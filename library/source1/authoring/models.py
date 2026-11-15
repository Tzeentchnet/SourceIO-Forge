from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ...source2.authoring.assets import ExtractedAuthoringModel
from ...source2.authoring.models import (
    AuthoringGrouping,
    AuthoringScope,
)
from ...source2.export import LossReport, ModelDocument
from ...source2.workshop import ContentRelativePath
from .target import Source1AuthoringTarget


_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or any(
            ord(character) < 32 for character in value
    ):
        raise ValueError(f"{label} must be non-empty text without control characters")
    return value


def _vector3(value, label: str) -> tuple[float, float, float]:
    normalized = tuple(float(component) for component in value)
    if len(normalized) != 3 or any(
            not math.isfinite(component) for component in normalized
    ):
        raise ValueError(f"{label} must contain three finite values")
    return normalized


class Source1ArtifactRoot(str, Enum):
    CONTENT = "content"
    GAME = "game"


class Source1ArtifactKind(str, Enum):
    MODEL_SOURCE = "model_source"
    MODEL_SCRIPT = "model_script"
    MATERIAL = "material"
    TEXTURE_SOURCE = "texture_source"
    TEXTURE_SETTINGS = "texture_settings"
    MANIFEST = "manifest"
    REPORT = "report"
    LOG = "log"


class Source1CompilerKind(str, Enum):
    VTEX = "vtex"
    STUDIOMDL = "studiomdl"


@dataclass(frozen=True, slots=True)
class Source1AuthoringCollision:
    document: ModelDocument
    mass: float | None = None
    concave: bool = True

    def __post_init__(self):
        if not isinstance(self.document, ModelDocument):
            raise TypeError("Source 1 collision geometry must be a ModelDocument")
        if self.mass is not None:
            mass = float(self.mass)
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError("Source 1 collision mass must be finite and positive")
            object.__setattr__(self, "mass", mass)
        if not isinstance(self.concave, bool):
            raise TypeError("Source 1 collision concavity must be a boolean")


@dataclass(frozen=True, slots=True)
class Source1AuthoringLod:
    distance: float
    document: ModelDocument

    def __post_init__(self):
        distance = float(self.distance)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Source 1 LOD distances must be finite and positive")
        if not isinstance(self.document, ModelDocument):
            raise TypeError("Source 1 LOD geometry must be a ModelDocument")
        object.__setattr__(self, "distance", distance)


@dataclass(frozen=True, slots=True)
class Source1BodygroupChoice:
    name: str
    document: ModelDocument | None

    def __post_init__(self):
        if not isinstance(self.name, str) or not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "Source 1 bodygroup choice names must contain only lowercase "
                "ASCII letters, digits, and underscores"
            )
        if self.document is not None and not isinstance(
                self.document,
                ModelDocument,
        ):
            raise TypeError("Source 1 bodygroup geometry must be a ModelDocument")


@dataclass(frozen=True, slots=True)
class Source1Bodygroup:
    name: str
    choices: tuple[Source1BodygroupChoice, ...]

    def __post_init__(self):
        _text(self.name, "Source 1 bodygroup name")
        choices = tuple(self.choices)
        if not choices or any(
                not isinstance(choice, Source1BodygroupChoice)
                for choice in choices
        ):
            raise ValueError(
                "Source 1 bodygroups require one or more typed choices"
            )
        names = tuple(choice.name.casefold() for choice in choices)
        if len(set(names)) != len(names):
            raise ValueError("Source 1 bodygroup choice names must be unique")
        object.__setattr__(self, "choices", choices)


@dataclass(frozen=True, slots=True)
class Source1SkinFamily:
    materials: tuple[str, ...]

    def __post_init__(self):
        materials = tuple(self.materials)
        if not materials:
            raise ValueError("Source 1 skin families cannot be empty")
        for material in materials:
            _text(material, "Source 1 skin material name")
            if (
                    "/" in material
                    or "\\" in material
                    or material.casefold().endswith(".vmt")
            ):
                raise ValueError(
                    "Source 1 skin materials must be bare VMT names without "
                    "an extension"
                )
        object.__setattr__(self, "materials", materials)


@dataclass(frozen=True, slots=True)
class Source1Attachment:
    name: str
    bone: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        _text(self.name, "Source 1 attachment name")
        _text(self.bone, "Source 1 attachment bone")
        object.__setattr__(
            self,
            "position",
            _vector3(self.position, "Source 1 attachment position"),
        )
        object.__setattr__(
            self,
            "rotation",
            _vector3(self.rotation, "Source 1 attachment rotation"),
        )


@dataclass(frozen=True, slots=True)
class Source1Hitbox:
    group: int
    bone: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self):
        if not isinstance(self.group, int) or self.group < 0:
            raise ValueError(
                "Source 1 hitbox groups must be non-negative integers"
            )
        _text(self.bone, "Source 1 hitbox bone")
        minimum = _vector3(self.minimum, "Source 1 hitbox minimum")
        maximum = _vector3(self.maximum, "Source 1 hitbox maximum")
        if any(lower > upper for lower, upper in zip(minimum, maximum)):
            raise ValueError(
                "Source 1 hitbox minimum must not exceed its maximum"
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@dataclass(frozen=True, slots=True)
class Source1Bone:
    name: str
    parent: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]

    def __post_init__(self):
        _text(self.name, "Source 1 bone name")
        if not isinstance(self.parent, int) or self.parent < -1:
            raise ValueError("Source 1 bone parents must be integer indices or -1")
        object.__setattr__(
            self,
            "position",
            _vector3(self.position, "Source 1 bone position"),
        )
        object.__setattr__(
            self,
            "rotation",
            _vector3(self.rotation, "Source 1 bone rotation"),
        )


@dataclass(frozen=True, slots=True)
class Source1Skeleton:
    bones: tuple[Source1Bone, ...]

    def __post_init__(self):
        bones = tuple(self.bones)
        if not bones or any(not isinstance(bone, Source1Bone) for bone in bones):
            raise ValueError(
                "Source 1 skeletons require one or more typed bones"
            )
        names = tuple(bone.name.casefold() for bone in bones)
        if len(set(names)) != len(names):
            raise ValueError("Source 1 skeleton bone names must be unique")
        for index, bone in enumerate(bones):
            if bone.parent >= index:
                raise ValueError(
                    "Source 1 skeleton parents must precede their children"
                )
        object.__setattr__(self, "bones", bones)


@dataclass(frozen=True, slots=True)
class Source1ExtractedAuthoringModel(ExtractedAuthoringModel):
    collision: Source1AuthoringCollision | None = None
    lods: tuple[Source1AuthoringLod, ...] = ()
    bodygroups: tuple[Source1Bodygroup, ...] = ()
    skin_families: tuple[Source1SkinFamily, ...] = ()
    attachments: tuple[Source1Attachment, ...] = ()
    hitboxes: tuple[Source1Hitbox, ...] = ()
    skeleton: Source1Skeleton | None = None

    def __post_init__(self):
        ExtractedAuthoringModel.__post_init__(self)
        if self.collision is not None and not isinstance(
                self.collision, Source1AuthoringCollision
        ):
            raise TypeError(
                "Source 1 extracted collision data must be "
                "a Source1AuthoringCollision"
            )
        lods = tuple(self.lods)
        if any(not isinstance(lod, Source1AuthoringLod) for lod in lods):
            raise TypeError(
                "Source 1 extracted LODs must be Source1AuthoringLod values"
            )
        lods = tuple(sorted(lods, key=lambda item: item.distance))
        distances = tuple(lod.distance for lod in lods)
        if len(set(distances)) != len(distances):
            raise ValueError("Source 1 LOD distances must be unique")
        object.__setattr__(self, "lods", lods)
        for field_name, item_type in (
                ("bodygroups", Source1Bodygroup),
                ("skin_families", Source1SkinFamily),
                ("attachments", Source1Attachment),
                ("hitboxes", Source1Hitbox),
        ):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, item_type) for value in values):
                raise TypeError(
                    f"Source 1 {field_name.replace('_', ' ')} must contain "
                    f"{item_type.__name__} values"
                )
            object.__setattr__(self, field_name, values)
        bodygroup_names = tuple(
            bodygroup.name.casefold() for bodygroup in self.bodygroups
        )
        if len(set(bodygroup_names)) != len(bodygroup_names):
            raise ValueError("Source 1 bodygroup names must be unique")
        if self.skeleton is not None and not isinstance(
                self.skeleton,
                Source1Skeleton,
        ):
            raise TypeError("Source 1 skeleton must be Source1Skeleton or None")


@dataclass(frozen=True, slots=True)
class Source1AuthoringRequest:
    target: Source1AuthoringTarget
    asset_name: str
    model_source_directory: ContentRelativePath = field(
        default_factory=lambda: ContentRelativePath("modelsrc/sourceio")
    )
    material_source_directory: ContentRelativePath = field(
        default_factory=lambda: ContentRelativePath("materialsrc/models/sourceio")
    )
    model_directory: ContentRelativePath = field(
        default_factory=lambda: ContentRelativePath("models/sourceio")
    )
    material_directory: ContentRelativePath = field(
        default_factory=lambda: ContentRelativePath("materials/models/sourceio")
    )
    surface_property: str = "default"
    scope: AuthoringScope = AuthoringScope.SELECTED
    grouping: AuthoringGrouping = AuthoringGrouping.COMBINED
    overwrite: bool = False
    compile_after_export: bool = False
    export_skeleton: bool = False

    def __post_init__(self):
        if not isinstance(self.target, Source1AuthoringTarget):
            raise TypeError("target must be a Source1AuthoringTarget")
        if not isinstance(self.asset_name, str):
            raise TypeError("asset_name must be a string")
        for field_name in (
            "model_source_directory",
            "material_source_directory",
            "model_directory",
            "material_directory",
        ):
            object.__setattr__(
                self,
                field_name,
                ContentRelativePath(getattr(self, field_name)),
            )
        if isinstance(self.scope, str) and not isinstance(self.scope, AuthoringScope):
            object.__setattr__(self, "scope", AuthoringScope(self.scope))
        if isinstance(self.grouping, str) and not isinstance(
                self.grouping, AuthoringGrouping
        ):
            object.__setattr__(self, "grouping", AuthoringGrouping(self.grouping))
        if not isinstance(self.surface_property, str):
            raise TypeError("surface_property must be a string")
        if not isinstance(self.export_skeleton, bool):
            raise TypeError("export_skeleton must be a boolean")


@dataclass(frozen=True, slots=True)
class Source1AuthoringArtifact:
    root: Source1ArtifactRoot
    path: ContentRelativePath
    kind: Source1ArtifactKind
    payload: bytes | str

    def __post_init__(self):
        if isinstance(self.root, str) and not isinstance(self.root, Source1ArtifactRoot):
            object.__setattr__(self, "root", Source1ArtifactRoot(self.root))
        object.__setattr__(self, "path", ContentRelativePath(self.path))
        if isinstance(self.kind, str) and not isinstance(self.kind, Source1ArtifactKind):
            object.__setattr__(self, "kind", Source1ArtifactKind(self.kind))
        if not isinstance(self.payload, (bytes, str)):
            raise TypeError("Source 1 authoring artifact payloads must be bytes or text")

    def encoded(self) -> bytes:
        return self.payload.encode("utf-8") if isinstance(self.payload, str) else self.payload

    @property
    def size(self) -> int:
        return len(self.encoded())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


@dataclass(frozen=True, slots=True)
class Source1CompileOutput:
    root: Source1ArtifactRoot
    path: ContentRelativePath

    def __post_init__(self):
        if isinstance(self.root, str) and not isinstance(self.root, Source1ArtifactRoot):
            object.__setattr__(self, "root", Source1ArtifactRoot(self.root))
        object.__setattr__(self, "path", ContentRelativePath(self.path))


@dataclass(frozen=True, slots=True)
class Source1CompileStep:
    compiler: Source1CompilerKind
    source_root: Source1ArtifactRoot
    source: ContentRelativePath
    expected_outputs: tuple[Source1CompileOutput, ...]
    arguments: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.compiler, str) and not isinstance(
                self.compiler, Source1CompilerKind
        ):
            object.__setattr__(self, "compiler", Source1CompilerKind(self.compiler))
        if isinstance(self.source_root, str) and not isinstance(
                self.source_root, Source1ArtifactRoot
        ):
            object.__setattr__(
                self,
                "source_root",
                Source1ArtifactRoot(self.source_root),
            )
        object.__setattr__(self, "source", ContentRelativePath(self.source))
        object.__setattr__(
            self,
            "expected_outputs",
            tuple(
                output
                if isinstance(output, Source1CompileOutput)
                else Source1CompileOutput(*output)
                for output in self.expected_outputs
            ),
        )
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not self.expected_outputs:
            raise ValueError("Source 1 compile steps require expected outputs")
        keys = {
            (output.root, output.path)
            for output in self.expected_outputs
        }
        if len(keys) != len(self.expected_outputs):
            raise ValueError("Source 1 compile outputs must be unique")
        if any(not isinstance(argument, str) for argument in self.arguments):
            raise TypeError("Source 1 compiler arguments must be strings")


@dataclass(slots=True)
class Source1BackendOutput:
    artifacts: tuple[Source1AuthoringArtifact, ...]
    compile_steps: tuple[Source1CompileStep, ...]
    model_paths: tuple[ContentRelativePath, ...]
    report: LossReport = field(default_factory=LossReport)

    def __post_init__(self):
        self.artifacts = tuple(self.artifacts)
        self.compile_steps = tuple(self.compile_steps)
        self.model_paths = tuple(ContentRelativePath(path) for path in self.model_paths)

    @classmethod
    def create(
            cls,
            artifacts: Iterable[Source1AuthoringArtifact],
            compile_steps: Iterable[Source1CompileStep],
            model_paths: Iterable[ContentRelativePath],
            *,
            report: LossReport | None = None,
    ) -> Source1BackendOutput:
        return cls(
            tuple(artifacts),
            tuple(compile_steps),
            tuple(model_paths),
            report if report is not None else LossReport(),
        )
