"""Standalone Source 1 SMD/VTA version-1 document support."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO, overload

from ...shared.vector_types import Vector2, Vector3

__all__ = [
    "SMDDocument",
    "SMDNode",
    "SMDBone",
    "SMDWeight",
    "SMDVertex",
    "SMDTriangle",
    "SMDVertexAnimationVertex",
    "parse",
    "dump",
    "load",
]


@dataclass(slots=True)
class SMDNode:
    id: int
    name: str
    parent: int


@dataclass(slots=True)
class SMDBone:
    bone: int
    position: Vector3[float]
    rotation: Vector3[float]


@dataclass(slots=True)
class SMDWeight:
    bone: int
    weight: float


@dataclass(slots=True)
class SMDVertex:
    bone: int
    position: Vector3[float]
    normal: Vector3[float]
    uv: Vector2[float]
    # None omits the optional link count; () writes an explicit zero link count.
    weights: tuple[SMDWeight, ...] | None = None


@dataclass(slots=True)
class SMDTriangle:
    material: str
    vertices: tuple[SMDVertex, SMDVertex, SMDVertex]


@dataclass(slots=True)
class SMDVertexAnimationVertex:
    index: int
    position: Vector3[float]
    normal: Vector3[float]


@dataclass(slots=True)
class SMDDocument:
    nodes: list[SMDNode] = field(default_factory=list)
    skeleton: dict[int, list[SMDBone]] = field(default_factory=dict)
    triangles: list[SMDTriangle] = field(default_factory=list)
    vertex_animation: dict[int, list[SMDVertexAnimationVertex]] = field(
        default_factory=dict
    )
    version: int = 1


_INTEGER_TOKEN = re.compile(r"[+-]?\d+")
_NUMBER_TOKEN = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)
_NODE_LINE = re.compile(r'(\d+)\s+"((?:[^"\\]|\\["\\])*)"\s+(-?\d+)')
_SECTIONS = frozenset(("nodes", "skeleton", "triangles", "vertexanimation"))
_REQUIRED_SECTIONS = frozenset(("nodes", "skeleton"))
_WEIGHT_SUM_TOLERANCE = 1e-6


def _strip_comment(line: str) -> str:
    quoted = False
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
        elif quoted and character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif (
            not quoted
            and character == "/"
            and line[index:index + 2] == "//"
            and (index == 0 or line[index - 1].isspace())
        ):
            return line[:index]
    return line


def _unescape_name(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\":
            index += 1
            character = value[index]
        result.append(character)
        index += 1
    return "".join(result)


def _escape_name(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _validate_string(value: str, description: str) -> None:
    if not isinstance(value, str) or any(ord(character) < 32 for character in value):
        raise ValueError(f"{description} cannot contain control characters")


def _validate_name(value: str) -> None:
    _validate_string(value, "SMD node name")


def _validate_material(value: str) -> None:
    _validate_string(value, "SMD triangle material")
    if (
        not value
        or value == "end"
        or value.strip() != value
        or _strip_comment(value).strip() != value
    ):
        raise ValueError("Invalid SMD triangle material")


class _Reader:
    def __init__(self, text: str):
        self._lines = iter(enumerate(text.removeprefix("\ufeff").splitlines(), 1))
        self.line = 0

    def next(self) -> str | None:
        for number, raw_line in self._lines:
            self.line = number
            line = _strip_comment(raw_line).strip()
            if line:
                return line
        return None

    def required(self, section: str) -> str:
        line = self.next()
        if line is None:
            raise ValueError(f"Unterminated SMD {section} section")
        return line

    def integer(self, token: str) -> int:
        if _INTEGER_TOKEN.fullmatch(token) is None:
            raise ValueError(f"Invalid integer {token!r} on SMD line {self.line}")
        return int(token)

    def number(self, token: str) -> float:
        try:
            result = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid number {token!r} on SMD line {self.line}"
            ) from exc
        if not math.isfinite(result):
            raise ValueError(f"Non-finite number on SMD line {self.line}")
        if _NUMBER_TOKEN.fullmatch(token) is None:
            raise ValueError(f"Invalid number {token!r} on SMD line {self.line}")
        return result

    def numbers(self, tokens: list[str], count: int) -> tuple[float, ...]:
        if len(tokens) != count:
            raise ValueError(
                f"Expected {count} values on SMD line {self.line}, got {len(tokens)}"
            )
        return tuple(self.number(token) for token in tokens)

    def vector2(self, tokens: list[str]) -> Vector2[float]:
        x, y = self.numbers(tokens, 2)
        return x, y

    def vector3(self, tokens: list[str]) -> Vector3[float]:
        x, y, z = self.numbers(tokens, 3)
        return x, y, z

    def read_nodes(self, document: SMDDocument) -> None:
        while (line := self.required("nodes")) != "end":
            match = _NODE_LINE.fullmatch(line)
            if match is None:
                raise ValueError(f"Invalid SMD node on line {self.line}")
            node_id, escaped_name, parent = match.groups()
            name = _unescape_name(escaped_name)
            _validate_name(name)
            document.nodes.append(
                SMDNode(self.integer(node_id), name, self.integer(parent))
            )

    def read_skeleton(self, document: SMDDocument) -> None:
        frame: list[SMDBone] | None = None
        while (line := self.required("skeleton")) != "end":
            tokens = line.split()
            if tokens[0] == "time":
                if len(tokens) != 2:
                    raise ValueError(f"Invalid SMD time on line {self.line}")
                time = self.integer(tokens[1])
                if time < 0 or time in document.skeleton:
                    raise ValueError(
                        f"Invalid or duplicate SMD frame {time} on line {self.line}"
                    )
                frame = document.skeleton[time] = []
                continue
            if frame is None or len(tokens) != 7:
                raise ValueError(f"Invalid SMD skeleton pose on line {self.line}")
            frame.append(
                SMDBone(
                    self.integer(tokens[0]),
                    self.vector3(tokens[1:4]),
                    self.vector3(tokens[4:7]),
                )
            )

    def read_vertex(self, line: str) -> SMDVertex:
        tokens = line.split()
        if len(tokens) < 9:
            raise ValueError(f"Invalid SMD triangle vertex on line {self.line}")
        weights: tuple[SMDWeight, ...] | None = None
        if len(tokens) > 9:
            count = self.integer(tokens[9])
            if count < 0 or len(tokens) != 10 + count * 2:
                raise ValueError(
                    f"Invalid SMD vertex link count on line {self.line}"
                )
            weights = tuple(
                SMDWeight(
                    self.integer(tokens[10 + index * 2]),
                    self.number(tokens[11 + index * 2]),
                )
                for index in range(count)
            )
            _validate_weight_values(weights, f"SMD line {self.line}")
        return SMDVertex(
            self.integer(tokens[0]),
            self.vector3(tokens[1:4]),
            self.vector3(tokens[4:7]),
            self.vector2(tokens[7:9]),
            weights,
        )

    def read_triangles(self, document: SMDDocument) -> None:
        while (material := self.required("triangles")) != "end":
            _validate_material(material)
            vertices = (
                self.read_vertex(self.required("triangles")),
                self.read_vertex(self.required("triangles")),
                self.read_vertex(self.required("triangles")),
            )
            document.triangles.append(SMDTriangle(material, vertices))

    def read_vertexanimation(self, document: SMDDocument) -> None:
        frame: list[SMDVertexAnimationVertex] | None = None
        while (line := self.required("vertexanimation")) != "end":
            tokens = line.split()
            if tokens[0] == "time":
                if len(tokens) != 2:
                    raise ValueError(
                        f"Invalid SMD vertexanimation time on line {self.line}"
                    )
                time = self.integer(tokens[1])
                if time < 0 or time in document.vertex_animation:
                    raise ValueError(
                        "Invalid or duplicate SMD vertexanimation frame "
                        f"{time} on line {self.line}"
                    )
                frame = document.vertex_animation[time] = []
                continue
            if frame is None or len(tokens) != 7:
                raise ValueError(
                    f"Invalid SMD vertexanimation vertex on line {self.line}"
                )
            index = self.integer(tokens[0])
            if index < 0:
                raise ValueError(
                    f"Invalid SMD vertexanimation index on line {self.line}"
                )
            frame.append(
                SMDVertexAnimationVertex(
                    index,
                    self.vector3(tokens[1:4]),
                    self.vector3(tokens[4:7]),
                )
            )


def _validate_integer(value: int, description: str, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{description} must be at least {minimum}")


def _finite_number(value: float, description: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{description} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite")
    return number


def _validate_vector(value: object, count: int, description: str) -> None:
    try:
        if len(value) != count:  # type: ignore[arg-type]
            raise ValueError(f"{description} must have {count} components")
        components = iter(value)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ValueError(f"{description} must have {count} components") from exc
    for component in components:
        _finite_number(component, description)


def _validate_weight_values(
    weights: tuple[SMDWeight, ...], description: str
) -> None:
    seen_bones: set[int] = set()
    values: list[float] = []
    for weight in weights:
        if not isinstance(weight, SMDWeight):
            raise ValueError(f"{description} has an invalid SMD weight")
        _validate_integer(weight.bone, f"{description} weight bone", minimum=0)
        if weight.bone in seen_bones:
            raise ValueError(f"{description} has duplicate SMD weight bones")
        seen_bones.add(weight.bone)
        value = _finite_number(weight.weight, f"{description} weight")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{description} has an invalid SMD weight")
        values.append(value)
    if math.fsum(values) > 1.0 + _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{description} SMD weights exceed one")


def _validate(document: SMDDocument) -> None:
    if not isinstance(document, SMDDocument):
        raise TypeError("Expected an SMDDocument")
    _validate_integer(document.version, "SMD version")
    if document.version != 1:
        raise ValueError("Only SMD version 1 is supported")

    by_id: dict[int, int] = {}
    for node in document.nodes:
        if not isinstance(node, SMDNode):
            raise ValueError("Invalid SMD node")
        _validate_integer(node.id, "SMD node id", minimum=0)
        _validate_integer(node.parent, "SMD node parent")
        _validate_name(node.name)
        if node.id in by_id:
            raise ValueError(f"Duplicate SMD node id {node.id}")
        by_id[node.id] = node.parent

    node_ids = set(by_id)
    for node_id, parent in by_id.items():
        if parent != -1 and parent not in node_ids:
            raise ValueError(f"SMD node {node_id} has unknown parent {parent}")

    checked: set[int] = set()
    for node_id in node_ids:
        ancestors: set[int] = set()
        current = node_id
        while current != -1 and current not in checked:
            if current in ancestors:
                raise ValueError("Cyclic SMD node hierarchy")
            ancestors.add(current)
            current = by_id[current]
        checked.update(ancestors)

    for time, frame in document.skeleton.items():
        _validate_integer(time, "SMD skeleton frame time", minimum=0)
        seen_bones: set[int] = set()
        for pose in frame:
            if not isinstance(pose, SMDBone):
                raise ValueError(f"Invalid SMD skeleton pose at time {time}")
            _validate_integer(pose.bone, "SMD skeleton bone", minimum=0)
            if pose.bone not in node_ids:
                raise ValueError(f"Unknown SMD skeleton bone {pose.bone}")
            if pose.bone in seen_bones:
                raise ValueError(f"Duplicate SMD skeleton bone {pose.bone}")
            seen_bones.add(pose.bone)
            _validate_vector(pose.position, 3, "SMD bone position")
            _validate_vector(pose.rotation, 3, "SMD bone rotation")

    for triangle in document.triangles:
        if not isinstance(triangle, SMDTriangle):
            raise ValueError("Invalid SMD triangle")
        _validate_material(triangle.material)
        if len(triangle.vertices) != 3:
            raise ValueError("SMD triangles must contain exactly three vertices")
        for vertex in triangle.vertices:
            if not isinstance(vertex, SMDVertex):
                raise ValueError("Invalid SMD triangle vertex")
            _validate_integer(vertex.bone, "SMD triangle bone", minimum=0)
            if vertex.bone not in node_ids:
                raise ValueError(f"Unknown SMD triangle bone {vertex.bone}")
            _validate_vector(vertex.position, 3, "SMD vertex position")
            _validate_vector(vertex.normal, 3, "SMD vertex normal")
            _validate_vector(vertex.uv, 2, "SMD vertex UV")
            if vertex.weights is not None:
                if not isinstance(vertex.weights, tuple):
                    raise ValueError("SMD vertex weights must be a tuple or None")
                _validate_weight_values(vertex.weights, "SMD vertex")
                for weight in vertex.weights:
                    if weight.bone not in node_ids:
                        raise ValueError(
                            f"Unknown SMD triangle weight bone {weight.bone}"
                        )

    for time, frame in document.vertex_animation.items():
        _validate_integer(time, "SMD vertexanimation frame time", minimum=0)
        seen_indices: set[int] = set()
        for vertex in frame:
            if not isinstance(vertex, SMDVertexAnimationVertex):
                raise ValueError(
                    f"Invalid SMD vertexanimation vertex at time {time}"
                )
            _validate_integer(
                vertex.index, "SMD vertexanimation vertex index", minimum=0
            )
            if vertex.index in seen_indices:
                raise ValueError(
                    f"Duplicate SMD vertexanimation index {vertex.index}"
                )
            seen_indices.add(vertex.index)
            _validate_vector(
                vertex.position, 3, "SMD vertexanimation position"
            )
            _validate_vector(vertex.normal, 3, "SMD vertexanimation normal")


def parse(text: str) -> SMDDocument:
    """Parse version-1 SMD or VTA text into a validated document."""
    if not isinstance(text, str):
        raise TypeError("SMD input must be text")
    reader = _Reader(text)
    if reader.next() != "version 1":
        raise ValueError("SMD input must start with version 1")

    document = SMDDocument()
    seen: set[str] = set()
    while (section := reader.next()) is not None:
        if section not in _SECTIONS or section in seen:
            raise ValueError(
                f"Unknown or duplicate SMD section {section!r} "
                f"on line {reader.line}"
            )
        seen.add(section)
        if section == "nodes":
            reader.read_nodes(document)
        elif section == "skeleton":
            reader.read_skeleton(document)
        elif section == "triangles":
            reader.read_triangles(document)
        else:
            reader.read_vertexanimation(document)

    missing = _REQUIRED_SECTIONS - seen
    if missing:
        raise ValueError(
            "SMD input is missing required section(s): "
            + ", ".join(sorted(missing))
        )
    _validate(document)
    return document


def _number(value: float) -> str:
    return str(_finite_number(value, "SMD value"))


@overload
def dump(document: SMDDocument, stream: None = None) -> str:
    ...


@overload
def dump(document: SMDDocument, stream: TextIO) -> None:
    ...


def dump(document: SMDDocument, stream: TextIO | None = None) -> str | None:
    """Write deterministic version-1 text, or return it when no stream is supplied."""
    _validate(document)
    lines = ["version 1", "nodes"]
    for node in document.nodes:
        lines.append(f'{node.id} "{_escape_name(node.name)}" {node.parent}')

    lines.extend(("end", "skeleton"))
    for time, frame in sorted(document.skeleton.items()):
        lines.append(f"time {time}")
        for pose in frame:
            values = (*pose.position, *pose.rotation)
            lines.append(f"{pose.bone} " + " ".join(map(_number, values)))
    lines.append("end")

    if document.triangles:
        lines.append("triangles")
        for triangle in document.triangles:
            lines.append(triangle.material)
            for vertex in triangle.vertices:
                values = (*vertex.position, *vertex.normal, *vertex.uv)
                line = f"{vertex.bone} " + " ".join(map(_number, values))
                if vertex.weights is not None:
                    line += f" {len(vertex.weights)}"
                    for weight in vertex.weights:
                        line += f" {weight.bone} {_number(weight.weight)}"
                lines.append(line)
        lines.append("end")

    if document.vertex_animation:
        lines.append("vertexanimation")
        for time, frame in sorted(document.vertex_animation.items()):
            lines.append(f"time {time}")
            for vertex in frame:
                values = (*vertex.position, *vertex.normal)
                lines.append(
                    f"{vertex.index} " + " ".join(map(_number, values))
                )
        lines.append("end")

    result = "\n".join(lines) + "\n"
    if stream is None:
        return result
    stream.write(result)
    return None


def load(
    filepath: str | Path | None = None,
    file: TextIO | BinaryIO | None = None,
) -> SMDDocument:
    """Load SMD/VTA from exactly one path or caller-owned stream."""
    if (filepath is None) == (file is None):
        raise ValueError("Provide exactly one of filepath or file")
    if filepath is not None:
        with Path(filepath).open("rb") as stream:
            data = stream.read()
    else:
        assert file is not None
        data = file.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8-sig")
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("SMD streams must return text or bytes")
    return parse(text)
