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
    Quadrupole,
    RFCavity,
    Sextupole,
)


@pytest.fixture
def defaults():
    return {"energy": 3e9, "harmonic_number": 936}


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


def test_elegant_element_from_string_handles_drift(defaults):
    drift = "drift,l=0.0450000"
    assert elegant_element_from_string("d1", drift, defaults).equals(
        Drift("d1", 0.0450000)
    )


def test_elegant_element_from_string_handles_arithmetic(defaults):
    drift = 'drift,l="0.04 2 /"'
    assert elegant_element_from_string("d1", drift, defaults).equals(Drift("d1", 0.02))


def test_elegant_element_from_string_handles_marker(defaults):
    marker = "mark"
    expected = Marker("m1")
    assert elegant_element_from_string("m1", marker, defaults).equals(expected)


def test_elegant_element_from_string_handles_quadrupole(defaults):
    quad = "KQUAD, N_KICKS=30, L=0.4064, K1=-0.7008"
    quad = quad.lower()
    expected = Quadrupole(
        "q1", 0.4064, k=-0.7008, NumIntSteps=30, PassMethod="StrMPoleSymplectic4Pass",
    )
    assert elegant_element_from_string("q1", quad, defaults).equals(expected)


def test_elegant_element_from_string_handles_sextupole(defaults):
    sext = "KSEXT,N_KICKS=12, L=0.29, K2=39.55"
    sext = sext.lower()
    # Note that h is divided by two due to Elegant's scaling.
    expected = Sextupole("s1", 0.29, h=19.775, NumIntSteps=12)
    assert elegant_element_from_string("s1", sext, defaults).equals(expected)


def test_elegant_element_from_string_handles_cavity(defaults):
    cavity = "RFCA, L=0.0, VOLT=2.5e6, FREQ=499654000, PHASE=156.7"
    cavity = cavity.lower()
    expected = RFCavity("c1", 0.0, 2.5e6, 4.99654e8, 936, 3e9, Phi="156.7")
    constructed = elegant_element_from_string("c1", cavity, defaults)
    assert constructed.equals(expected)


def test_elegant_element_from_string_handles_bending(defaults):
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
    assert elegant_element_from_string("b1", bend, defaults).equals(expected)


def test_elegant_element_from_string_handles_variable(defaults):
    drift = "drift,l=a"
    defaults["a"] = 1
    correct_drift = Drift("d1", 1)
    constructed_drift = elegant_element_from_string("d1", drift, defaults)
    assert correct_drift.equals(constructed_drift)


def test_expand_elegant():
    contents = """dmult:drift,l=1
diad6d:line=(dmult)"""
    elements = expand_elegant(contents, "diad6d", 3e9, 936)
    assert len(elements) == 1
    assert elements[0].equals(Drift("dmult", 1.0))
