"""
Module to define common elements used in AT.

Each element has a default PassMethod attribute for which it should have the
appropriate attributes.  If a different PassMethod is set, it is the caller's
responsibility to ensure that the appropriate attributes are present.
"""
import numpy
import copy
import itertools


def _array(value, shape=(-1,), dtype=numpy.float64):
    # Ensure proper ordering(F) and alignment(A) for "C" access in integrators
    return numpy.require(value, dtype=dtype, requirements=['F', 'A']).reshape(
        shape)


def _array66(value):
    return _array(value, shape=(6, 6))


def _nop(value):
    return value


class Element(object):
    """Base of pyat elements"""
    REQUIRED_ATTRIBUTES = ['FamName']
    CONVERSIONS = dict(R1=_array66, R2=_array66, T1=lambda v: _array(v, (6,)),
                       T2=lambda v: _array(v, (6,)),
                       RApertures=lambda v: _array(v, (4,)),
                       EApertures=lambda v: _array(v, (2,)),
                       Energy=float,
                       )

    entrance_fields = ['T1', 'R1', 'EntranceAngle', 'FringeInt1',
                       'FringeBendEntrance', 'FringeQuadEntrance']
    exit_fields = ['T2', 'R2', 'ExitAngle', 'FringeInt2',
                   'FringeBendExit', 'FringeQuadExit']

    def __init__(self, family_name, Length=0.0, PassMethod='IdentityPass',
                 **kwargs):
        self.FamName = family_name
        self.Length = float(Length)
        self.PassMethod = PassMethod

        for (key, value) in kwargs.items():
            try:
                setattr(self, key, self.CONVERSIONS.get(key, _nop)(value))
            except Exception as exc:
                exc.args = (
                    'In element {0}, parameter {1}: {2}'.format(
                        family_name, key, exc),)
                raise

    def __str__(self):
        first3 = ['FamName', 'Length', 'PassMethod']
        keywords = ['{0} : {1!s}'.format(k, self.__dict__[k]) for k in first3]
        keywords = keywords + ['{0} : {1!s}'.format(k, v) for k, v in
                               vars(self).items() if k not in first3]
        return '\n'.join((self.__class__.__name__ + ':', '\n'.join(keywords)))

    def __repr__(self):
        def differ(v1, v2):
            if isinstance(v1, numpy.ndarray):
                return not numpy.array_equal(v1, v2)
            else:
                return v1 != v2

        defelem = self.__class__(
            *(getattr(self, k) for k in self.REQUIRED_ATTRIBUTES))
        arguments = ('{0!r}'.format(getattr(self, k)) for k in
                     self.REQUIRED_ATTRIBUTES)
        keywords = ('{0}={1!r}'.format(k, v) for k, v in vars(self).items()
                    if differ(v, getattr(defelem, k, None)))
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(
            itertools.chain(arguments, keywords)))

    def copy(self):
        """Return a shallow copy of the element"""
        return copy.copy(self)


class LongElement(Element):
    """pyAT long element"""

    def divide(self, frac, keep_axis=False):
        """split the element in len(frac) pieces whose length
        is frac[i]*self.Length

        arguments:
            frac            length of each slice expressed as a fraction of the
                            initial length. sum(frac) may differ from 1.

        keywords:
            keep_axis=False If True, displacement and rotation are applied to
                            each slice, if False they are applied
                            at extremities only

        Return a list of elements equivalent to the original.

        Example:

        >>> Drift('dr', 0.5).divide([0.2, 0.6, 0.2])
        [Drift('dr', 0.1), Drift('dr', 0.3), Drift('dr', 0.1)]
        """
        def popattr(element, attr):
            val = getattr(element, attr)
            delattr(element, attr)
            return attr, val

        def part(fr):
            pp = el.copy()
            pp.Length = fr * el.Length
            if hasattr(el, 'BendingAngle'):
                pp.BendingAngle = fr / numpy.sum(frac) * el.BendingAngle
            return pp

        frac = numpy.asarray(frac, dtype=float)
        el = self.copy()
        first = 0 if keep_axis else 2
        # Remove entrance and exit attributes
        fin = dict(popattr(el, key) for key in vars(self) if
                   key in self.entrance_fields[first:])
        fout = dict(popattr(el, key) for key in vars(self) if
                    key in self.exit_fields[first:])
        # Split element
        element_list = [part(f) for f in frac]
        # Restore entrance and exit attributes
        for key, value in fin.items():
            setattr(element_list[0], key, value)
        for key, value in fout.items():
            setattr(element_list[-1], key, value)
        return element_list


class Marker(Element):
    """pyAT marker element"""

    def __init__(self, family_name, **kwargs):
        super(Marker, self).__init__(family_name, **kwargs)


class Monitor(Element):
    """pyAT monitor element"""

    def __init__(self, family_name, **kwargs):
        super(Monitor, self).__init__(family_name, **kwargs)


class Aperture(Element):
    """pyAT aperture element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Limits']
    CONVERSIONS = dict(Element.CONVERSIONS, Limits=lambda v: _array(v, (4,)))

    def __init__(self, family_name, limits, **kwargs):
        kwargs.setdefault('PassMethod', 'AperturePass')
        super(Aperture, self).__init__(family_name, Limits=limits, **kwargs)


class Drift(LongElement):
    """pyAT drift space element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length']

    def __init__(self, family_name, length, **kwargs):
        """Drift(FamName, Length, **keywords)
        """
        kwargs.setdefault('PassMethod', 'DriftPass')
        super(Drift, self).__init__(family_name,
                                    Length=kwargs.pop('Length', length),
                                    **kwargs)

    def insert(self, insert_list):
        """insert elements inside a drift

        arguments:
            insert_list iterable. Each item of elems is iself an iterable
                        with 2 objects
                        The 1st object is the location of the elements to
                        be inserted, given as a fraction of the Drift length.
                        The 2nd object is an element to be inserted at that
                        location. If None, self will be divided but no element
                        will be inserted.

        Return a list of elements.

        Drifts with negative lengths may be generated if necessary.

        Examples:

        >>> Drift('dr', 2.0).insert(((0.25, None), (0.75, None)))
        [Drift('dr', 0.5), Drift('dr', 1.0), Drift('dr', 0.5)]

        >>> Drift('dr', 2.0).insert(((0.0, Marker('m1')), (0.5, Marker('m2'))))
        [Marker('m1'), Drift('dr', 1.0), Marker('m2'), Drift('dr', 1.0)]

        >>> Drift('dr', 2.0).insert(((0.5, Quadrupole('qp', 0.4, 0.0)),))
        [Drift('dr', 0.8), Quadrupole('qp', 0.4), Drift('dr', 0.8)]
        """
        frac, elements = zip(*insert_list)
        lg = [0.0 if el is None else el.Length for el in elements]
        fr = numpy.asarray(frac, dtype=float)
        lg = 0.5 * numpy.asarray(lg, dtype=float) / self.Length
        drfrac = numpy.hstack((fr-lg, 1.0)) - numpy.hstack((0.0, fr+lg))
        long_elems = (drfrac != 0.0)
        drifts = numpy.ndarray((len(drfrac),), dtype='O')
        drifts[long_elems] = self.divide(drfrac[long_elems])
        line = [None]*(len(drifts)+len(elements))
        line[::2] = drifts
        line[1::2] = elements
        return [el for el in line if el is not None]


class ThinMultipole(Element):
    """pyAT thin multipole element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['PolynomA',
                                                         'PolynomB']
    CONVERSIONS = dict(Element.CONVERSIONS, BendingAngle=float, MaxOrder=int,
                       PolynomB=_array, PolynomA=_array)

    def __init__(self, family_name, poly_a, poly_b, MaxOrder=0, **kwargs):
        """ThinMultipole(FamName, PolynomA, PolynomB, **keywords)

        Available keywords:
        MaxOrder      Number of desired multipoles
        """
        kwargs.setdefault('PolynomA', poly_a)
        kwargs.setdefault('PolynomB', poly_b)
        kwargs.setdefault('PassMethod', 'ThinMPolePass')
        super(ThinMultipole, self).__init__(family_name, MaxOrder=MaxOrder,
                                            **kwargs)
        # Adjust polynom lengths
        poly_size = max(self.MaxOrder + 1, len(self.PolynomA),
                        len(self.PolynomB))
        self.PolynomA = numpy.concatenate(
            (self.PolynomA, numpy.zeros(poly_size - len(self.PolynomA))))
        self.PolynomB = numpy.concatenate(
            (self.PolynomB, numpy.zeros(poly_size - len(self.PolynomB))))


class Multipole(LongElement, ThinMultipole):
    """pyAT multipole element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length',
                                                         'PolynomA',
                                                         'PolynomB']
    CONVERSIONS = dict(ThinMultipole.CONVERSIONS, NumIntSteps=int,
                       K=float, KickAngle=_array)

    def __init__(self, family_name, length, poly_a, poly_b, NumIntSteps=10,
                 **kwargs):
        """Multipole(FamName, Length, PolynomA, PolynomB, **keywords)

        Available keywords:
        MaxOrder        Number of desired multipoles
        NumIntSteps     Number of integration steps (default: 10)
        """
        kwargs.setdefault('PassMethod', 'StrMPoleSymplectic4Pass')
        super(Multipole, self).__init__(family_name, poly_a, poly_b,
                                        Length=kwargs.pop('Length', length),
                                        NumIntSteps=NumIntSteps, **kwargs)


class Dipole(Multipole):
    """pyAT dipole element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length',
                                                         'BendingAngle',
                                                         'K']
    CONVERSIONS = dict(Multipole.CONVERSIONS, EntranceAngle=float,
                       ExitAngle=float,
                       FringeQuadEntrance=int, FringeQuadExit=int,
                       FringeBendEntrance=int, FringeBendExit=int)

    def __init__(self, family_name, length, BendingAngle, k=0.0,
                 EntranceAngle=0.0, ExitAngle=0.0, **kwargs):
        """Dipole(FamName, Length, BendingAngle, Strength=0, **keywords)

        Available keywords:
        EntranceAngle entrance angle (default 0.0)
        ExitAngle       exit angle (default 0.0)
        PolynomB        straight multipoles
        PolynomA        skew multipoles
        MaxOrder        Number of desired multipoles
        NumIntSteps     Number of integration steps (default: 10)
        FullGap
        FringeInt1
        FringeInt2
        FringeBendEntrance
        FringeBendExit
        FringeQuadEntrance
        FringeQuadExit
        fringeIntM0
        fringeIntP0
         KickAngle
   """
        poly_b = kwargs.pop('PolynomB', numpy.array([0, k]))
        _ = kwargs.pop('K', None)
        kwargs.setdefault('PassMethod', 'BendLinearPass')
        super(Dipole, self).__init__(family_name, length, [], poly_b,
                                     BendingAngle=BendingAngle,
                                     EntranceAngle=EntranceAngle,
                                     ExitAngle=ExitAngle, **kwargs)

    @property
    def K(self):
        return self.PolynomB[1]

    @K.setter
    def K(self, strength):
        self.PolynomB[1] = strength


# Bend is a synonym of Dipole.
Bend = Dipole


class Quadrupole(Multipole):
    """pyAT quadrupole element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length', 'K']
    CONVERSIONS = dict(Multipole.CONVERSIONS, FringeQuadEntrance=int,
                       FringeQuadExit=int)

    def __init__(self, family_name, length, k=0.0, MaxOrder=1, **kwargs):
        """Quadrupole(FamName, Length, Strength=0, **keywords)

        Available keywords:
        PolynomB        straight multipoles
        PolynomA        skew multipoles
        MaxOrder        Number of desired multipoles
        NumIntSteps     Number of integration steps (default: 10)
        FringeQuadEntrance
        FringeQuadExit
        fringeIntM0
        fringeIntP0
        KickAngle
        """
        poly_b = kwargs.pop('PolynomB', numpy.array([0, k]))
        _ = kwargs.pop('K', None)
        kwargs.setdefault('PassMethod', 'QuadLinearPass')
        super(Quadrupole, self).__init__(family_name, length, [], poly_b,
                                         MaxOrder=MaxOrder, **kwargs)

    @property
    def K(self):
        return self.PolynomB[1]

    @K.setter
    def K(self, strength):
        self.PolynomB[1] = strength


class Sextupole(Multipole):
    """pyAT sextupole element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length']

    def __init__(self, family_name, length, h=0.0, MaxOrder=2, **kwargs):
        """Sextupole(FamName, Length, Strength=0, **keywords)

        Available keywords:
        PolynomB        straight multipoles
        PolynomA        skew multipoles
        MaxOrder        Number of desired multipoles
        NumIntSteps     Number of integration steps (default: 10)
        """
        poly_b = kwargs.pop('PolynomB', [0, 0, h])
        kwargs.setdefault('PassMethod', 'StrMPoleSymplectic4Pass')
        super(Sextupole, self).__init__(family_name, length, [], poly_b,
                                        MaxOrder=MaxOrder, **kwargs)


class Octupole(Multipole):
    """pyAT octupole element, with no changes from multipole at present"""
    REQUIRED_ATTRIBUTES = Multipole.REQUIRED_ATTRIBUTES


class RFCavity(Element):
    """pyAT RF cavity element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length', 'Voltage',
                                                         'Frequency',
                                                         'HarmNumber', 'Energy']
    CONVERSIONS = dict(Element.CONVERSIONS, Voltage=float, Frequency=float,
                       HarmNumber=int, TimeLag=float)

    def __init__(self, family_name, length, voltage, frequency, harmonic_number,
                 energy, TimeLag=0.0, **kwargs):
        """
        Available keywords:
        TimeLag   time lag with respect to the reference particle
        """
        kwargs.setdefault('PassMethod', 'CavityPass')
        super(RFCavity, self).__init__(family_name, length, Voltage=voltage,
                                       Frequency=frequency,
                                       HarmNumber=harmonic_number,
                                       Energy=energy, TimeLag=TimeLag, **kwargs)


class RingParam(Element):
    """pyAT RingParam element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Energy']
    CONVERSIONS = dict(Element.CONVERSIONS, Energy=float, Periodicity=int)

    def __init__(self, family_name, energy, Periodicity=1, **kwargs):
        kwargs.setdefault('PassMethod', 'IdentityPass')
        super(RingParam, self).__init__(family_name, Energy=energy,
                                        Periodicity=Periodicity, **kwargs)


class M66(Element):
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES
    CONVERSIONS = dict(Element.CONVERSIONS, M66=_array66)

    def __init__(self, family_name, m66=None, **kwargs):
        if m66 is None:
            m66 = numpy.identity(6)
        kwargs.setdefault('PassMethod', 'Matrix66Pass')
        super(M66, self).__init__(family_name, M66=m66, **kwargs)


class Corrector(LongElement):
    """pyAT corrector element"""
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length', 'KickAngle']
    CONVERSIONS = dict(Element.CONVERSIONS, KickAngle=_array)

    def __init__(self, family_name, length, kick_angle, **kwargs):
        kwargs.setdefault('PassMethod', 'CorrectorPass')
        super(Corrector, self).__init__(family_name,
                                        kwargs.pop('Length', length),
                                        KickAngle=kick_angle, **kwargs)
