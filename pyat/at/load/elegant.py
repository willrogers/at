import collections
from os.path import abspath
import re
import numpy

from at.lattice.elements import (
    Drift,
    Dipole,
    Quadrupole,
    Sextupole,
    Multipole,
    Corrector,
    Marker,
    RFCavity,
)
from at.lattice import Lattice
from at.load import register_format


ELEMENT_MAP = {
    "drift": Drift,
    "drif": Drift,
    "csben": Dipole,
    "csbend": Dipole,
    "csrcsben": Dipole,
    "quadrupole": Quadrupole,
    "kquad": Quadrupole,
    "sextupole": Sextupole,
    "ksext": Sextupole,
    "multipole": Multipole,
    "corrector": Corrector,
    "kicker": Corrector,
    "mark": Marker,
    "malign": Marker,
    "recirc": Marker,
    "sreffects": Marker,
    "rcol": Marker,
    "watch": Marker,
    "charge": Marker,
    "monitor": Marker,
    "cavity": RFCavity,
    "rfca": RFCavity,
}


def split_ignoring_parentheses(string, delimiter):
    PLACEHOLDER = "placeholder"
    substituted = string[:]
    matches = collections.deque(re.finditer("\(.*?\)", string))
    for match in matches:
        substituted = "{}{}{}".format(
            substituted[: match.start()], PLACEHOLDER, substituted[match.end() :]
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


def create_element(cls, name, params, energy, harmonic_number):
    length = params.pop("l", 0)
    if "n_kicks" in params:
        params["NumIntSteps"] = params.pop("n_kicks")
    if cls == Marker:
        return cls(name, **params)
    elif cls == Dipole:
        params["PassMethod"] = "BndMPoleSymplectic4Pass"
        params["BendingAngle"] = float(params.pop("angle"))
        params["EntranceAngle"] = float(params.pop("e1"))
        params["ExitAngle"] = float(params.pop("e2"))
        params["FullGap"] = float(params.pop("hgap")) * 2
        fint = params.pop("fint")
        params["FringeInt1"] = fint
        params["FringeInt2"] = fint
        k1 = float(params.pop("k1", 0))
        k2 = float(params.pop("k2", 0))
        k3 = float(params.pop("k3", 0))
        k4 = float(params.pop("k4", 0))
        params["PolynomB"] = [0, k1, k2, k3, k4]
        return cls(name, length, **params)
    elif cls == Quadrupole:
        params["k"] = float(params.pop("k1"))
        params["PassMethod"] = "StrMPoleSymplectic4Pass"
        return cls(name, length, **params)
    elif cls == RFCavity:
        voltage = params.pop("volt")
        frequency = params.pop("freq")
        params["Phi"] = params.pop("phase")
        return cls(name, length, voltage, frequency, harmonic_number, energy, **params)
    elif cls == Sextupole:
        k2 = float(params.pop("k2", 0))
        return cls(name, length, k2, **params)
    elif cls == Multipole:
        k1 = float(params.pop("k1", 0))
        k2 = float(params.pop("k2", 0))
        poly_a = [0, 0, 0, 0]
        poly_b = [0, k1, k2, 0]
        return cls(name, length, poly_a, poly_b, **params)
    elif cls == Corrector:
        hkick = params.pop("hkick")
        vkick = params.pop("vkick")
        kick_angle = [hkick, vkick]
        return cls(name, length, kick_angle, **params)
    else:
        return cls(name, length, **params)


def parse_lines(contents):
    """Return individual lines.

    Remove comments and whitespace, convert to lowercase, and split on
    semicolons.
    """
    lines = [l.lower().strip() for l in contents.splitlines()]
    parsed_lines = []
    current_line = ""
    for l in lines:
        if not l or l.startswith("!"):
            continue
        if l.endswith("&"):
            current_line += l[:-1]
        else:
            parsed_lines.append(current_line + l)
            current_line = ""

    return parsed_lines


def parse_chunk(value, elements, chunks):
    chunk = []
    parts = split_ignoring_parentheses(value, ",")
    for part in parts:
        if "symmetry" in part:
            continue
        if "line" in part:
            line_parts = re.match("line=\((.*)\)", part).groups()[0]
            for p in line_parts.split(","):
                p = p.strip()
                if p.startswith("-"):
                    chunk.extend(reversed(chunks[p[1:]]))
                elif p in elements:
                    chunk.append(elements[p.strip()])
                elif p in chunks:
                    chunk.extend(chunks[p.strip()])
                else:
                    raise Exception(f"What is {p}?")
        elif "inv" in part:
            chunk_to_invert = re.match("inv\((.*)\)", part).groups()[0]
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
        elif "*" in part:
            num, chunk_name = part.split("*")
            chunk.extend(int(num) * chunks[chunk_name])
        elif part in elements:
            chunk.append(elements[part])
        elif part in chunks:
            chunk.extend(chunks[part])
        else:
            raise Exception("part {} not understood".format(part))
    return chunk


def expand_elegant(contents, lattice_key, energy, harmonic_number):
    lines = parse_lines(contents)
    variables = {"energy": energy, "harmonic_number": harmonic_number}
    elements = {}
    chunks = {}
    for line in lines:
        if ":" not in line:
            key, value = line.split("=")
            variables[key.strip()] = value.strip()
        else:
            key, value = line.split(":")
            key = key.strip()
            value = value.strip()
            if value.split(",")[0] in ELEMENT_MAP:
                elements[key] = elegant_element_from_string(key, value, variables)
            else:
                chunk = parse_chunk(value, elements, chunks)
                chunks[key] = chunk

    return chunks[lattice_key]


def handle_value(value):
    value = value.strip()
    if value.startswith('"'):
        assert value.endswith('"')
        value = value[1:-1]
        value = value.split()
        if len(value) > 1:
            assert len(value) == 3
            if value[2] == "/":
                value = float(value[0]) / float(value[1])
            elif value[2] == "-":
                value = float(value[0]) - float(value[1])
        else:
            value = value[0]
    return value


def elegant_element_from_string(name, element_string, variables):
    parts = split_ignoring_parentheses(element_string, ",")
    params = {}
    element_type = parts[0]
    for part in parts[1:]:
        key, value = split_ignoring_parentheses(part, "=")
        key = key.strip()
        value = handle_value(value)
        if value in variables:
            value = variables[value]
        else:
            params[key] = value
    if element_type == "sextupole":
        params["h"] = float(params.pop("k", 0))

    energy = variables["energy"]
    harmonic_number = variables["energy"]

    return create_element(
        ELEMENT_MAP[element_type], name, params, energy, harmonic_number
    )


def load_elegant(filename, **kwargs):
    energy = kwargs.pop("energy")
    lattice_key = kwargs.pop("lattice_key")
    harmonic_number = kwargs.pop("harmonic_number")

    def elem_iterator(params, tracy_file):
        with open(params.setdefault("tracy_file", tracy_file)) as f:
            contents = f.read()
            element_lines = expand_elegant(
                contents, lattice_key, energy, harmonic_number
            )
            params.setdefault("energy", energy)
            for line in element_lines:
                yield line

    return Lattice(abspath(filename), iterator=elem_iterator, **kwargs)


register_format(".lte", load_elegant, descr="Elegant format")
