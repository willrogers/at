from at.load.tracy import (
    expand_tracy, strip_comments, tracy_element_from_string
)
from at.lattice.elements import (
    Drift, Dipole, Marker, Quadrupole, RFCavity, Sextupole
)


def test_strip_comments_removes_comments():
    assert strip_comments('a{}b') == 'ab'


def test_strip_comments_removes_whitespace():
    assert strip_comments('a\n{a\nb}b') == 'ab'


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
    expected = Quadrupole('q1', 0.15, k=10.24, n=10, method='4')
    assert tracy_element_from_string('q1', quad, variables) == expected


def test_tracy_element_from_string_handles_sextupole():
    quad = 'sextupole,l=0.14,k=174.4,n=nsext,method=4'
    variables = {'nsext': 2}
    expected = Sextupole('s1', 0.14, h=174.4, n=2, method='4')
    assert tracy_element_from_string('s1', quad, variables) == expected


def test_tracy_element_from_string_handles_cavity():
    cavity = 'cavity,l=0.0,frequency=499.654e6,voltage=2.2e6,phi=0.0'
    expected = RFCavity('c1', 0.0, 2.2e6, 4.99654e8, 31, 3e9, phi='0.0')
    constructed = tracy_element_from_string('c1', cavity, {})
    assert constructed == expected


def test_tracy_element_from_string_handles_bending():
    bend = 'bending,l= 0.20000000,t=0.32969999,t1=0.00000000,t2=0.32969999,k=-0.12411107,n=nbend,method=4'
    variables = {'nbend': 2}
    expected = Dipole(
        'b1',
        0.2,
        BendingAngle=0.32969999,
        EntranceAngle=0,
        ExitAngle=0.32969999,
        k=-0.12411107,
        n=2,
        method='4'
    )
    assert tracy_element_from_string('b1', bend, variables) == expected


def test_tracy_element_from_string_handles_variable():
    drift = 'drift,l=a'
    variables = {'a': 1}
    correct_drift = Drift('d1', 1)
    constructed_drift = tracy_element_from_string('d1', drift, variables)
    assert correct_drift == constructed_drift


def test_expand_tracy():
    contents = "define lattice;dmult:drift,l=1;cell:dmult;end"
    elements = expand_tracy(contents)
    assert len(elements) == 1
    assert elements[0] == Drift('dmult', 1.0)
