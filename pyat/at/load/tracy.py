import collections
from os.path import abspath
import re
import numpy

from at.lattice.elements import (
    Drift, Dipole, Quadrupole, Sextupole, Multipole, Corrector, Marker, RFCavity
)
from at.lattice import Lattice
from at.load import register_format


ELEMENT_MAP = {
    'drift': Drift,
    'bending': Dipole,
    'quadrupole': Quadrupole,
    'sextupole': Sextupole,
    'multipole': Multipole,
    'corrector': Corrector,
    'marker': Marker,
    'beampositionmonitor': Marker,
    'cavity': RFCavity
}


def split_ignoring_parentheses(string, delimiter):
    PLACEHOLDER = 'placeholder'
    substituted = string[:]
    matches = collections.deque(re.finditer('\(.*?\)', string))
    for match in matches:
        substituted = '{}{}{}'.format(
            substituted[:match.start()],
            PLACEHOLDER,
            substituted[match.end():]
        )
    parts = substituted.split(delimiter)
    replaced_parts = []
    for part in parts:
        if PLACEHOLDER in part:
            next_match = matches.popleft()
            part = part.replace(PLACEHOLDER, next_match.group())
        replaced_parts.append(part)
    assert not matches

    return replaced_parts


def create_element(cls, name, params, energy):
    if 'n' in params:
        params['NumIntSteps'] = params.pop('n')
    if cls == Marker:
        return cls(name, **params)
    elif cls == Dipole:
        length = params.pop('l')
        params['PassMethod'] = 'BndMPoleSymplectic4Pass'
        params['BendingAngle'] = (float(params.pop('t')) / 180) * numpy.pi
        params['EntranceAngle'] = (float(params.pop('t1')) / 180) * numpy.pi
        params['ExitAngle'] = (float(params.pop('t2')) / 180) * numpy.pi
        return cls(name, length, **params)
    elif cls == Quadrupole:
        length = params.pop('l')
        params['PassMethod'] = 'StrMPoleSymplectic4Pass'
        return cls(name, length, **params)
    elif cls == RFCavity:
        length = params.pop('l')
        voltage = params.pop('voltage')
        frequency = params.pop('frequency')
        # Harmonic number not in Tracy lattice
        h = 31
        return cls(name, length, voltage, frequency, h, energy*1e9, **params)
    elif cls == Multipole:
        length = params.pop('l')
        poly_a = [0, 0, 0, 0]
        poly_b = [0, 0, 0, 0]
        return cls(name, length, poly_a, poly_b, **params)

    else:
        length = params.pop('l', 0)
        return cls(name, length, **params)


def parse_lines(contents):
    """Return individual lines.

    Remove comments and whitespace, convert to lowercase, and split on
    semicolons.
    """
    # Nested comments not handled.
    in_comment = False
    stripped_contents = ''
    for char in contents.lower():
        if char == '{':
            in_comment = True
        elif char == '}':
            in_comment = False
        elif not in_comment and not char.isspace():
            stripped_contents += char
    return stripped_contents.split(';')


def parse_chunk(value, elements, chunks):
    chunk = []
    parts = split_ignoring_parentheses(value, ',')
    for part in parts:
        if 'symmetry' in part:
            continue
        if 'inv' in part:
            chunk_to_invert = re.match('inv\((.*)\)', part).groups()[0]
            inverted_chunk = []
            for el in reversed(chunks[chunk_to_invert]):
                if el.__class__ == Dipole:
                    inverted_dipole = el.copy()
                    inverted_dipole.EntranceAngle = el.ExitAngle
                    inverted_dipole.ExitAngle = el.EntranceAngle
                    inverted_chunk.append(inverted_dipole)
                else:
                    inverted_chunk.append(el)
            chunk.extend(inverted_chunk)
        elif '*' in part:
            num, chunk_name = part.split('*')
            chunk.extend(int(num) * chunks[chunk_name])
        elif part in elements:
            chunk.append(elements[part])
        elif part in chunks:
            chunk.extend(chunks[part])
        else:
            raise Exception('part {} not understood'.format(part))
    return chunk


def expand_tracy(contents):
    lines = parse_lines(contents)
    assert lines[0] == 'definelattice'
    assert lines[-2] == 'end'
    assert lines[-1] == ''
    variables = {}
    elements = {}
    chunks = {}
    for line in lines[1:-2]:
        if ':' not in line:
            key, value = line.split('=')
            variables[key] = value
        else:
            key, value = line.split(':')
            if value.split(',')[0].strip() in ELEMENT_MAP:
                elements[key] = tracy_element_from_string(key, value, variables)
            else:
                chunk = parse_chunk(value, elements, chunks)
                chunks[key] = chunk

    return chunks['cell'], float(variables['energy']) * 1e9


def tracy_element_from_string(name, element_string, variables):
    parts = split_ignoring_parentheses(element_string, ',')
    params = {}
    element_type = parts[0]
    if element_type == 'corrector':
        parts = parts[:1]
        params['kick_angle'] = 0
    for part in parts[1:]:
        key, value = split_ignoring_parentheses(part, '=')
        if value in variables:
            value = variables[value]
        if key == 'hom':
            assert value[0] == '('
            assert value[-1] == ')'
            order, a, b = value[1:-1].split(',')
            polynom_a = [0] * int(order)
            polynom_b = [0] * int(order)
            polynom_a[int(order)-1] = float(a)
            polynom_b[int(order)-1] = float(b)
            params['PolynomA'] = polynom_a
            params['PolynomB'] = polynom_b
        else:
            params[key] = value
    if element_type == 'sextupole':
        params['h'] = float(params.pop('k', 0))

    try:
        energy = float(variables['energy'])
    except KeyError:
        energy = None

    return create_element(ELEMENT_MAP[element_type], name, params, energy)


def load_tracy(filename, **kwargs):
    def elem_iterator(params, tracy_file):
        with open(params.setdefault('tracy_file', tracy_file)) as f:
            contents = f.read()
            element_lines, energy = expand_tracy(contents)
            params.setdefault('energy', energy)
            for line in element_lines:
                yield line

    return Lattice(abspath(filename), iterator=elem_iterator, **kwargs)


register_format('.lat', load_tracy, descr='Tracy format')