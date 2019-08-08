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


def create_element(cls, name, params, energy):
    if 'n' in params:
        params['NumIntSteps'] = params.pop('n')
    if cls == Marker:
        return cls(name, **params)
    elif cls == Dipole:
        length = params.pop('l')
        params['PassMethod'] = 'BndMPoleSymplectic4Pass'
        params['BendingAngle'] = float(params.pop('t')) / 360 * numpy.pi
        params['EntranceAngle'] = float(params.pop('t1')) / 360 * numpy.pi
        params['ExitAngle'] = float(params.pop('t2')) / 360 * numpy.pi
        return cls(name, length, **params)
    elif cls == Quadrupole:
        length = params.pop('l')
        params['PassMethod'] = 'StrMPoleSymplectic4Pass'
        return cls(name, length, **params)

    elif cls == Sextupole:
        params['h'] = params.pop('k')
        length = params.pop('l', 0)
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


def extract_raw(contents):
    """Remove comments and whitespace and convert to lowercase."""
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
    return stripped_contents


def parse_chunk(value, elements, chunks):
    chunk = []
    parts = value.split(',')
    for part in parts:
        if 'symmetry' in part:
            continue
        if 'inv' in part:
            inverted_chunk = re.match('inv\((.*)\)', part).groups()[0]
            chunk.extend(reversed(chunks[inverted_chunk]))
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
    stripped_contents = extract_raw(contents)
    lines = stripped_contents.split(';')
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

    return chunks['cell']


def tracy_element_from_string(name, element_string, variables):
    parts = element_string.split(',')
    params = {}
    element_type = parts[0]
    if element_type == 'corrector':
        parts = parts[:1]
        params['kick_angle'] = 0
    for part in parts[1:]:
        key, value = part.split('=')
        if value in variables:
            value = variables[value]
        params[key] = value

    energy = float(variables['energy'])

    return create_element(ELEMENT_MAP[element_type], name, params, energy)


def load_tracy(filename, **kwargs):
    def elem_iterator(params, tracy_file):
        with open(tracy_file) as f:
            contents = f.read()
            element_lines = expand_tracy(contents)
            for line in element_lines:
                yield line

    return Lattice(abspath(filename), iterator=elem_iterator, **kwargs)


register_format('.lat', load_tracy, descr='Tracy format')