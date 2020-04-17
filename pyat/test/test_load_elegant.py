import pytest

from at.load.elegant import (
    expand_elegant,
    parse_lines,
    elegant_element_from_string,
    split_ignoring_parentheses,
)
from at.lattice.elements import (
    Dipole,
    Drift,
    Marker,
    Multipole,
    Quadrupole,
    RFCavity,
    Sextupole,
)


@pytest.mark.parametrize(
    "string,delimiter,target",
    [
        ["a,b", ",", ["a", "b"]],
        ["a,b(c,d)", ",", ["a", "b(c,d)"]],
        ["l=0,hom(4,0.0,0)", ",", ["l=0", "hom(4,0.0,0)"]],
    ],
)
def test_split_ignoring_parentheses(string, delimiter, target):
    assert split_ignoring_parentheses(string, delimiter) == target


def test_parse_lines_removes_comments():
    assert parse_lines("a\n!b\nc") == ["a", "c"]


def test_parse_lines_handles_ampersand():
    assert parse_lines("a&\nb") == ["ab"]
    assert parse_lines("a&\nb\nc") == ["ab", "c"]


def test_elegant_element_from_string_handles_drift():
    drift = "drift,l=0.0450000"
    assert elegant_element_from_string("d1", drift, {}) == Drift("d1", 0.0450000)


def test_elegant_element_from_string_handles_marker():
    marker = "mark"
    expected = Marker("m1")
    assert elegant_element_from_string("m1", marker, {}) == expected


def test_elegant_element_from_string_handles_quadrupole():
    quad = "KQUAD, N_KICKS=30, L=0.4064, K1=-0.7008"
    quad = quad.lower()
    expected = Quadrupole(
        "q1", 0.4064, k=-0.7008, NumIntSteps=30, PassMethod="StrMPoleSymplectic4Pass",
    )
    assert elegant_element_from_string("q1", quad, {}) == expected


def test_elegant_element_from_string_handles_sextupole():
    sext = "KSEXT,N_KICKS=12, L=0.29, K2= 39.55"
    sext = sext.lower()
    expected = Sextupole("s1", 0.29, h=39.55, NumIntSteps=12)
    assert elegant_element_from_string("s1", sext, {}) == expected


def test_elegant_element_from_string_handles_hom():
    oct = "multipole,l=0.0,hom=(4,1.0,0.3)"
    expected = Multipole("m1", 0.0, poly_a=[0, 0, 0, 1], poly_b=[0, 0, 0, 0.3])
    assert elegant_element_from_string("m1", oct, {}) == expected


def test_elegant_element_from_string_handles_cavity():
    cavity = "RFCA, L=0.0, VOLT=2.5e6, FREQ=499654000, PHASE=156.7"
    cavity = cavity.lower()
    expected = RFCavity("c1", 0.0, 2.5e6, 4.99654e8, 31, 3.5e9, Phi="156.7")
    constructed = elegant_element_from_string("c1", cavity, {"energy": 3.5})
    assert constructed == expected


def test_elegant_element_from_string_handles_bending():
    bend = "CSBEN,L=0.933,K1=0,Angle=0.1308,E1=0.06544,E2=0.06544, N_KICKS=50, HGAP=0.0233, FINT=0.6438"
    bend = bend.lower()
    expected = Dipole(
        "b1",
        0.933,
        BendingAngle=0.1308,
        EntranceAngle=0.06544,
        ExitAngle=0.06544,
        FringeInt1=0.6438,
        FringeInt2=0.6438,
        FullGap=0.0466,
        k=0,
        NumIntSteps=50,
        PassMethod="BndMPoleSymplectic4Pass",
        PolynomA=[0, 0, 0, 0, 0],
        PolynomB=[0, 0, 0, 0, 0],
    )
    assert elegant_element_from_string("b1", bend, {}) == expected


def test_elegant_element_from_string_handles_variable():
    drift = "drift,l=a"
    variables = {"a": 1}
    correct_drift = Drift("d1", 1)
    constructed_drift = elegant_element_from_string("d1", drift, variables)
    assert correct_drift == constructed_drift


def test_expand_elegant():
    contents = """dmult:drift,l=1
diad6d:line=(dmult)"""
    elements, energy = expand_elegant(contents)
    assert len(elements) == 1
    assert elements[0] == Drift("dmult", 1.0)
