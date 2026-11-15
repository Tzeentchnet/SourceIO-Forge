from __future__ import annotations

import math
from dataclasses import dataclass

from ...source2.authoring import AuthoringMaterial
from ...source2.workshop import ContentRelativePath


def _optional_finite(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class Source1PhongParameters:
    exponent: float | None = None
    boost: float | None = None
    tint: tuple[float, float, float] | None = None
    fresnel_ranges: tuple[float, float, float] | None = None
    albedo_tint: bool = False
    rim_light: bool = False
    rim_exponent: float | None = None
    rim_boost: float | None = None

    def __post_init__(self):
        exponent = _optional_finite(self.exponent, "Phong exponent")
        boost = _optional_finite(self.boost, "Phong boost")
        rim_exponent = _optional_finite(
            self.rim_exponent,
            "Rim-light exponent",
        )
        rim_boost = _optional_finite(self.rim_boost, "Rim-light boost")
        for label, value in (
                ("Phong exponent", exponent),
                ("Phong boost", boost),
                ("Rim-light exponent", rim_exponent),
                ("Rim-light boost", rim_boost),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{label} cannot be negative")
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "boost", boost)
        object.__setattr__(self, "rim_exponent", rim_exponent)
        object.__setattr__(self, "rim_boost", rim_boost)
        if not isinstance(self.albedo_tint, bool):
            raise TypeError("Phong albedo tint must be a boolean")
        if not isinstance(self.rim_light, bool):
            raise TypeError("Rim-light state must be a boolean")
        if not self.rim_light and (
                rim_exponent is not None or rim_boost is not None
        ):
            raise ValueError(
                "Rim-light exponent and boost require rim lighting"
            )
        for field_name, width in (("tint", 3), ("fresnel_ranges", 3)):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = tuple(float(component) for component in value)
            if len(normalized) != width or any(
                    not math.isfinite(component) for component in normalized
            ):
                raise ValueError(
                    f"Phong {field_name.replace('_', ' ')} must contain "
                    f"{width} finite values"
                )
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True, slots=True)
class Source1MaterialParameters:
    self_illum: bool = False
    self_illum_tint: tuple[float, float, float] | None = None
    envmap: str | None = None
    envmap_tint: tuple[float, float, float] | None = None
    envmap_contrast: float | None = None
    envmap_saturation: float | None = None

    def __post_init__(self):
        if not isinstance(self.self_illum, bool):
            raise TypeError("Self-illumination state must be a boolean")
        if self.self_illum_tint is not None and not self.self_illum:
            raise ValueError(
                "Self-illumination tint requires self-illumination"
            )
        envmap = self.envmap
        if envmap is not None:
            if (
                    not isinstance(envmap, str)
                    or not envmap
                    or envmap.startswith(("/", "\\"))
                    or ".." in envmap.replace("\\", "/").split("/")
                    or envmap.casefold().endswith(".vtf")
                    or any(ord(character) < 32 for character in envmap)
            ):
                raise ValueError(
                    "Source 1 envmaps must be safe material-relative names "
                    "without a .vtf extension"
                )
            object.__setattr__(self, "envmap", envmap.replace("\\", "/"))
        for field_name in ("self_illum_tint", "envmap_tint"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = tuple(float(component) for component in value)
            if len(normalized) != 3 or any(
                    not math.isfinite(component) for component in normalized
            ):
                raise ValueError(
                    f"{field_name.replace('_', ' ')} must contain three "
                    "finite values"
                )
            object.__setattr__(self, field_name, normalized)
        for field_name in ("envmap_contrast", "envmap_saturation"):
            value = _optional_finite(
                getattr(self, field_name),
                field_name.replace("_", " "),
            )
            if value is not None and value < 0:
                raise ValueError(
                    f"{field_name.replace('_', ' ')} cannot be negative"
                )
            object.__setattr__(self, field_name, value)
        if self.envmap is None and any(
                value is not None
                for value in (
                    self.envmap_tint,
                    self.envmap_contrast,
                    self.envmap_saturation,
                )
        ):
            raise ValueError("Envmap parameters require an envmap")


def source1_pbr_to_phong(
        roughness: float,
        metallic: float,
        *,
        max_exponent: float = 150.0,
        dielectric_boost: float = 0.25,
) -> Source1PhongParameters:
    roughness = float(roughness)
    metallic = float(metallic)
    max_exponent = float(max_exponent)
    dielectric_boost = float(dielectric_boost)
    if (
            not math.isfinite(roughness)
            or not 0.0 <= roughness <= 1.0
            or not math.isfinite(metallic)
            or not 0.0 <= metallic <= 1.0
    ):
        raise ValueError("PBR roughness and metallic values must be from zero to one")
    if not math.isfinite(max_exponent) or not 1.0 <= max_exponent <= 150.0:
        raise ValueError("PBR-to-phong maximum exponent must be from 1 to 150")
    if (
            not math.isfinite(dielectric_boost)
            or not 0.0 <= dielectric_boost <= 1.0
    ):
        raise ValueError("PBR-to-phong dielectric boost must be from zero to one")
    alpha = max(roughness * roughness, 1e-4)
    exponent = min(max(2.0 / (alpha * alpha) - 2.0, 1.0), max_exponent)
    boost = dielectric_boost + (1.0 - dielectric_boost) * metallic
    return Source1PhongParameters(
        exponent=exponent,
        boost=boost,
        albedo_tint=metallic > 0.0,
    )


@dataclass(frozen=True, slots=True)
class Source1AuthoringMaterial(AuthoringMaterial):
    existing_vmt: ContentRelativePath | None = None
    phong: Source1PhongParameters | None = None
    parameters: Source1MaterialParameters | None = None
    pbr_converted: bool = False

    def __post_init__(self):
        AuthoringMaterial.__post_init__(self)
        if self.existing_resource is not None:
            raise ValueError(
                "Source 1 materials cannot reuse Source 2 material resources"
            )
        if self.existing_vmt is not None:
            resource = ContentRelativePath(self.existing_vmt)
            if (
                    resource.suffix.casefold() != ".vmt"
                    or resource.parts[0].casefold() == "materials"
            ):
                raise ValueError(
                    "Existing Source 1 materials must be material-relative "
                    f"editable .vmt paths: {resource}"
                )
            if (
                    self.textures
                    or self.phong is not None
                    or self.parameters is not None
                    or self.pbr_converted
            ):
                raise ValueError(
                    "Reused Source 1 materials cannot also author textures or "
                    "typed parameters"
                )
            object.__setattr__(self, "existing_vmt", resource)
        if self.phong is not None and not isinstance(
                self.phong,
                Source1PhongParameters,
        ):
            raise TypeError("phong must be Source1PhongParameters or None")
        if self.parameters is not None and not isinstance(
                self.parameters,
                Source1MaterialParameters,
        ):
            raise TypeError(
                "parameters must be Source1MaterialParameters or None"
            )
        if not isinstance(self.pbr_converted, bool):
            raise TypeError("pbr_converted must be a boolean")
        if self.pbr_converted and self.phong is None:
            raise ValueError("PBR conversion requires authored phong parameters")
