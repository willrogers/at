import pytest
import numpy
from at import atpass
from at import elements


@pytest.fixture
def rin():
    rin = numpy.array(numpy.zeros((1,6)))
    return rin


def test_correct_dimensions_does_not_raise_error(rin):
    l = []
    atpass(l, rin, 1)
    rin = numpy.zeros((6,))
    atpass(l, rin, 1)
    rin = numpy.zeros((2,6))


def test_incorrect_types_raises_value_error(rin):
    l = []
    with pytest.raises(ValueError):
        atpass(1, rin, 1)
    with pytest.raises(ValueError):
        atpass(l, 1, 1)
    with pytest.raises(ValueError):
        atpass(l, rin, 'a')


def test_incorrect_dimensions_raises_value_error():
    l = []
    rin = numpy.array(numpy.zeros((1,7)))
    with pytest.raises(ValueError):
        atpass(l, rin, 1)
    rin = numpy.array(numpy.zeros((6,1)))
    with pytest.raises(ValueError):
        atpass(l, rin, 1)


def test_fortran_aligned_array_raises_value_error():
    rin = numpy.asfortranarray(numpy.zeros((2,6)))
    l = []
    with pytest.raises(ValueError):
        atpass(l, rin, 1)


def test_missing_pass_method_raises_attribute_error(rin):
    m = elements.Marker('marker')
    l = [m]
    del m.PassMethod
    with pytest.raises(AttributeError):
        atpass(l, rin, 1)


def test_missing_length_raises_attribute_error(rin):
    m = elements.Drift('drift', 1.0)
    l = [m]
    del m.Length
    with pytest.raises(AttributeError):
        atpass(l, rin, 1)


@pytest.mark.parametrize("reuse", (True, False))
def test_reuse_attributes(rin, reuse):
    m = elements.Drift('drift', 1.0)
    l = [m]
    rin[0,0] = 1e-6
    rin[0,1] = 1e-6
    rin_copy = numpy.copy(rin)
    # two turns with original lattice
    atpass(l, rin, 2)
    # one turn with original lattice
    atpass(l, rin_copy, 1)
    # change an attribute
    m.Length = 2
    # one turn with altered lattice
    atpass(l, rin_copy, 1, reuse=reuse)
    if reuse:
        numpy.testing.assert_equal(rin, rin_copy)
    else:
        with pytest.raises(AssertionError):
            numpy.testing.assert_equal(rin, rin_copy)


@pytest.mark.parametrize("dipole_class", (elements.Dipole, elements.Bend))
def test_dipole(rin, dipole_class):
    print(elements.__file__)
    b = dipole_class('dipole', 1.0, 0.1, EntranceAngle=0.05, ExitAngle=0.05)
    l = [b]
    rin[0,0] = 1e-6
    rin_orig = numpy.copy(rin)
    atpass(l, rin, 1)
    rin_expected = numpy.array([1e-6, 0, 0, 0, 0, 1e-7]).reshape((1,6))
    numpy.testing.assert_almost_equal(rin_orig, rin_expected)


def test_marker(rin):
    m = elements.Marker('marker')
    assert m.Length == 0
    lattice = [m]
    rin = numpy.random.rand(*rin.shape)
    rin_orig = numpy.array(rin, copy=True)
    atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_aperture_inside_limits(rin):
    a = elements.Aperture('aperture', [-1e-3, 1e-3, -1e-4, 1e-4])
    assert a.Length == 0
    lattice = [a]
    rin[0][0] = 1e-5
    rin[0][2] = -1e-5
    rin_orig = numpy.array(rin, copy=True)
    atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_aperture_outside_limits(rin):
    a = elements.Aperture('aperture', [-1e-3, 1e-3, -1e-4, 1e-4])
    assert a.Length == 0
    lattice = [a]
    rin[0][0] = 1e-2
    rin[0][2] = -1e-2
    atpass(lattice, rin, 1)
    assert numpy.isinf(rin[0][0])
    assert rin[0][2] == -1e-2  # Only the first coordinate is marked as infinity


def test_drift_offset(rin):
    d = elements.Drift('drift', 1)
    lattice = [d]
    rin[0][0] = 1e-6
    rin[0][2] = 2e-6
    rin_orig = numpy.array(rin, copy=True)
    atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_drift_divergence(rin):
    d = elements.Drift('drift', 1.0)
    assert d.Length == 1
    lattice = [d]
    rin[0][1] = 1e-6
    rin[0][3] = -2e-6
    atpass(lattice, rin, 1)
    # results from Matlab
    rin_expected = numpy.array([1e-6, 1e-6, -2e-6, -2e-6, 0, 2.5e-12]).reshape(1,6)
    numpy.testing.assert_equal(rin, rin_expected)


def test_drift_two_particles(rin):
    d = elements.Drift('drift', 1.0)
    assert d.Length == 1
    lattice = [d]
    two_rin = numpy.concatenate((rin, rin), axis=0)
    # particle one is offset
    two_rin[0][0] = 1e-6
    two_rin[0][2] = 2e-6
    # particle two has divergence
    two_rin[1][1] = 1e-6
    two_rin[1][3] = -2e-6
    two_rin_orig = numpy.array(two_rin, copy=True)
    atpass(lattice, two_rin, 1)
    # results from Matlab
    p1_expected = numpy.array(two_rin_orig[0,:]).reshape(1,6)
    p2_expected = numpy.array([1e-6, 1e-6, -2e-6, -2e-6, 0, 2.5e-12]).reshape(1,6)
    two_rin_expected = numpy.concatenate((p1_expected, p2_expected), axis=0)
    numpy.testing.assert_equal(two_rin, two_rin_expected)


def test_quad(rin):
    q = elements.Quadrupole('quad', 0.4, k=1)
    lattice = [q]
    rin[0, 0] = 1e-6
    atpass(lattice, rin, 1)
    print(rin)
    expected = numpy.array([0.921060994002885,
                            -0.389418342308651,
                            0,
                            0,
                            0,
                            0.000000010330489]).reshape(1, 6) * 1e-6
    numpy.testing.assert_allclose(rin, expected)


def test_quad_incorrect_array(rin):
    q = elements.Quadrupole('quad', 0.4, k=1)
    q.PolynomB = 'a'
    lattice = [q]
    with pytest.raises(RuntimeError):
        atpass(lattice, rin, 1)
