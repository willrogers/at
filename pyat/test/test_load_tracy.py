import pytest

from at.load.tracy import (
    expand_tracy, parse_lines, tracy_element_from_string, split_ignoring_parentheses
)
from at.lattice.elements import (
    Dipole, Drift, Marker, Multipole, Quadrupole, RFCavity, Sextupole
)


@pytest.mark.parametrize('string,delimiter,target', [
    ['a,b', ',', ['a', 'b']],
    ['a,b(c,d)', ',', ['a', 'b(c,d)']],
    ['l=0,hom(4,0.0,0)', ',', ['l=0', 'hom(4,0.0,0)']],
])
def test_split_ignoring_parentheses(string, delimiter, target):
    assert split_ignoring_parentheses(string, delimiter) == target


def test_parse_lines_removes_comments():
    assert parse_lines('a{}b') == ['ab']


def test_parse_lines_removes_whitespace():
    assert parse_lines('a\n{a\nb}b') == ['ab']


def test_parse_line_splits_on_semicolons():
    assert parse_lines('a;b') == ['a', 'b']


def test_tracy_element_from_string_handles_drift():
    drift = 'drift,l=0.0450000'
    assert tracy_element_from_string('d1', drift, {}) == Drift('d1', 0.0450000)


def test_tracy_element_from_string_handles_marker():
    marker = 'marker'
    expected = Marker('m1')
    assert tracy_element_from_string('m1', marker, {}) == expected


def test_tracy_element_from_string_handles_quadrupole():
    quad = 'quadrupole,l=0.15,k=10.24,n=nquad,method=4'
    variables = {'nquad': 10}
    expected = Quadrupole(
        'q1',
        0.15,
        k=10.24,
        NumIntSteps=10,
        method='4',
        PassMethod='StrMPoleSymplectic4Pass'
    )
    assert tracy_element_from_string('q1', quad, variables) == expected


def test_tracy_element_from_string_handles_sextupole():
    sext = 'sextupole,l=0.14,k=174.4,n=nsext,method=4'
    variables = {'nsext': 2}
    expected = Sextupole('s1', 0.14, h=174.4, NumIntSteps=2, method='4')
    assert tracy_element_from_string('s1', sext, variables) == expected


def test_tracy_element_from_string_handles_hom():
    oct = 'multipole,l=0.0,hom=(4,1.0,0.3)'
    expected = Multipole('m1', 0.0, poly_a=[0, 0, 0, 1], poly_b=[0, 0, 0, 0.3])
    assert tracy_element_from_string('m1', oct, {}) == expected


def test_tracy_element_from_string_handles_cavity():
    cavity = 'cavity,l=0.0,frequency=499.654e6,voltage=2.2e6,phi=0.0'
    expected = RFCavity('c1', 0.0, 2.2e6, 4.99654e8, 31, 3.5e9, phi='0.0')
    constructed = tracy_element_from_string('c1', cavity, {'energy': 3.5})
    assert constructed == expected


def test_tracy_element_from_string_handles_bending():
    bend = 'bending,l= 0.20000000,t=0.32969999,t1=0.00000000,t2=0.32969999,k=-0.12411107,n=nbend,method=4'
    variables = {'nbend': 2}
    expected = Dipole(
        'b1',
        0.2,
        BendingAngle=0.00575435036929238,
        EntranceAngle=0,
        ExitAngle=0.00575435036929238,
        k=-0.12411107,
        NumIntSteps=2,
        method='4',
        PassMethod='BndMPoleSymplectic4Pass'
    )
    assert tracy_element_from_string('b1', bend, variables) == expected


def test_tracy_element_from_string_handles_variable():
    drift = 'drift,l=a'
    variables = {'a': 1}
    correct_drift = Drift('d1', 1)
    constructed_drift = tracy_element_from_string('d1', drift, variables)
    assert correct_drift == constructed_drift


def test_expand_tracy():
    contents = "define lattice;energy=1;dmult:drift,l=1;cell:dmult;end;"
    elements, energy = expand_tracy(contents)
    assert len(elements) == 1
    assert energy == 1e9
    assert elements[0] == Drift('dmult', 1.0)
