from __future__ import annotations

import json
import math
from typing import Iterable

from ...source2.authoring import AuthoringMaterial, MaterialAlphaMode, TextureRole
from ...source2.export import ModelDocument
from ...source2.workshop import ContentRelativePath
from .assets import Source1AuthoringMaterial
from .models import Source1Attachment, Source1Hitbox, Source1Skeleton


def _float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Source 1 authored numeric values must be finite")
    text = format(number, ".9g")
    return "0" if text == "-0" else text


def _quoted(value: str) -> str:
    if any(ord(character) < 32 for character in value):
        raise ValueError("Source 1 authored strings cannot contain control characters")
    return json.dumps(value, ensure_ascii=True)


def _texture_reference(path: ContentRelativePath) -> str:
    if not path.parts or path.parts[0].casefold() != "materialsrc":
        raise ValueError("Source texture paths must be below materialsrc")
    parts = list(path.parts[1:])
    suffix = ContentRelativePath("/".join(parts)).suffix
    if not suffix:
        raise ValueError("Source texture paths must have a file extension")
    parts[-1] = parts[-1][:-len(suffix)]
    return "/".join(parts)


def _smd_vector(value) -> tuple[float, float, float]:
    x, y, z = value
    return float(y), -float(x), float(z)


def _qc_vector(value) -> str:
    return " ".join(_float(component) for component in value)


def serialize_smd(
        document: ModelDocument,
        material_names: set[str] | frozenset[str],
        *,
        require_uv: bool = True,
        skeleton: Source1Skeleton | None = None,
) -> str:
    if skeleton is None:
        node_lines = ('  0 "root" -1',)
        pose_lines = ("  0 0 0 0 0 0 0",)
    else:
        node_lines = tuple(
            f"  {index} {_quoted(bone.name)} {bone.parent}"
            for index, bone in enumerate(skeleton.bones)
        )
        pose_lines = tuple(
            "  "
            + str(index)
            + " "
            + " ".join(
                _float(value)
                for value in (*bone.position, *bone.rotation)
            )
            for index, bone in enumerate(skeleton.bones)
        )
    lines = [
        "version 1",
        "nodes",
        *node_lines,
        "end",
        "skeleton",
        "time 0",
        *pose_lines,
        "end",
        "triangles",
    ]
    for mesh in sorted(document.meshes, key=lambda item: (item.name.casefold(), item.name)):
        for face in mesh.faces:
            if len(face.vertices) != 3:
                raise ValueError(
                    f"Source 1 SMD export requires triangles: {mesh.name}"
                )
            if face.material not in material_names:
                raise ValueError(
                    f"Source 1 SMD face references unknown material: {face.material}"
                )
            if any(ord(character) < 32 for character in face.material):
                raise ValueError("Source 1 SMD material names cannot contain control characters")
            lines.append(face.material)
            for vertex_index in face.vertices:
                vertex = mesh.vertices[vertex_index]
                normal = vertex.corner.normal
                if normal is None:
                    raise ValueError(
                        f"Source 1 SMD vertex has no normal: {mesh.name}"
                    )
                if require_uv and not vertex.corner.texcoords:
                    raise ValueError(
                        f"Source 1 SMD vertex has no UV map: {mesh.name}"
                    )
                texcoord = (
                    vertex.corner.texcoords[0]
                    if vertex.corner.texcoords
                    else (0.0, 0.0)
                )
                values = (
                    *_smd_vector(vertex.position),
                    *_smd_vector(normal),
                    texcoord[0],
                    1.0 - texcoord[1],
                )
                if skeleton is None:
                    lines.append(
                        "  0 "
                        + " ".join(_float(value) for value in values)
                    )
                    continue
                indices = tuple(vertex.corner.bone_indices)
                weights = tuple(vertex.corner.bone_weights)
                if not indices or len(indices) != len(weights):
                    raise ValueError(
                        f"Source 1 skeletal SMD vertex has invalid weights: "
                        f"{mesh.name}"
                    )
                combined: dict[int, float] = {}
                for bone_index, weight in zip(indices, weights):
                    bone_index = int(bone_index)
                    weight = float(weight)
                    if (
                            bone_index < 0
                            or bone_index >= len(skeleton.bones)
                            or not math.isfinite(weight)
                            or weight <= 0
                    ):
                        raise ValueError(
                            "Source 1 skeletal SMD vertex has an invalid bone "
                            f"link: {mesh.name}"
                        )
                    combined[bone_index] = (
                        combined.get(bone_index, 0.0) + weight
                    )
                total = sum(combined.values())
                links = tuple(sorted(
                    (
                        (bone_index, weight / total)
                        for bone_index, weight in combined.items()
                    ),
                    key=lambda item: (-item[1], item[0]),
                ))
                lines.append(
                    f"  {links[0][0]} "
                    + " ".join(_float(value) for value in values)
                    + f" {len(links)} "
                    + " ".join(
                        f"{bone_index} {_float(weight)}"
                        for bone_index, weight in links
                    )
                )
    lines.extend(("end", ""))
    return "\n".join(lines)


def serialize_qc(
        model_path: ContentRelativePath,
        smd_name: str,
        material_directory: ContentRelativePath,
        *,
        surface_property: str = "default",
        collision_smd_name: str | None = None,
        collision_mass: float | None = None,
        collision_concave: bool = True,
        lods: Iterable[tuple[float, str]] = (),
        bodygroups: Iterable[tuple[str, Iterable[str | None]]] = (),
        skin_families: Iterable[Iterable[str]] = (),
        attachments: Iterable[Source1Attachment] = (),
        hitboxes: Iterable[Source1Hitbox] = (),
        static_prop: bool = True,
) -> str:
    if model_path.parts[0].casefold() != "models":
        raise ValueError("Compiled Source 1 model paths must be below models")
    if material_directory.parts[0].casefold() != "materials":
        raise ValueError("Source 1 material directories must be below materials")
    model_reference = "/".join(model_path.parts[1:])
    material_reference = "/".join(material_directory.parts[1:])
    lod_entries = tuple(sorted(
        ((float(distance), str(name)) for distance, name in lods),
        key=lambda item: item[0],
    ))
    if any(not math.isfinite(distance) or distance <= 0 for distance, _ in lod_entries):
        raise ValueError("Source 1 LOD distances must be finite and positive")
    if len({distance for distance, _ in lod_entries}) != len(lod_entries):
        raise ValueError("Source 1 LOD distances must be unique")
    if collision_mass is not None:
        collision_mass = float(collision_mass)
        if not math.isfinite(collision_mass) or collision_mass <= 0:
            raise ValueError("Source 1 collision mass must be finite and positive")
    if collision_smd_name is None and (
            collision_mass is not None or collision_concave is not True
    ):
        raise ValueError("Source 1 collision settings require collision geometry")
    bodygroup_entries = tuple(
        (str(name), tuple(choices))
        for name, choices in bodygroups
    )
    if any(not name or not choices for name, choices in bodygroup_entries):
        raise ValueError(
            "Source 1 bodygroups require a name and at least one choice"
        )
    skin_entries = tuple(tuple(family) for family in skin_families)
    if skin_entries:
        width = len(skin_entries[0])
        if not width or any(len(family) != width for family in skin_entries):
            raise ValueError(
                "Source 1 skin families must be non-empty and have equal widths"
            )
    attachment_entries = tuple(attachments)
    if any(
            not isinstance(attachment, Source1Attachment)
            for attachment in attachment_entries
    ):
        raise TypeError("Source 1 QC attachments must be Source1Attachment values")
    hitbox_entries = tuple(hitboxes)
    if any(not isinstance(hitbox, Source1Hitbox) for hitbox in hitbox_entries):
        raise TypeError("Source 1 QC hitboxes must be Source1Hitbox values")

    lines = [
        f"$modelname {_quoted(model_reference)}",
        f"$body {_quoted('body')} {_quoted(smd_name)}",
        f"$surfaceprop {_quoted(surface_property)}",
        f"$cdmaterials {_quoted(material_reference)}",
    ]
    if static_prop:
        lines.insert(2, "$staticprop")
    for name, choices in bodygroup_entries:
        lines.append(f"$bodygroup {_quoted(name)}")
        lines.append("{")
        for choice in choices:
            lines.append(
                "    blank"
                if choice is None
                else f"    studio {_quoted(str(choice))}"
            )
        lines.append("}")
    if skin_entries:
        lines.append(f"$texturegroup {_quoted('skinfamilies')}")
        lines.append("{")
        for family in skin_entries:
            lines.append(
                "    { "
                + " ".join(_quoted(str(material)) for material in family)
                + " }"
            )
        lines.append("}")
    for distance, lod_smd_name in lod_entries:
        lines.extend((
            f"$lod {_float(distance)}",
            "{",
            (
                f"    replacemodel {_quoted(smd_name)} "
                f"{_quoted(lod_smd_name)}"
            ),
            "}",
        ))
    if collision_smd_name is not None:
        lines.extend((
            f"$collisionmodel {_quoted(collision_smd_name)}",
            "{",
            (
                f"    $mass {_float(collision_mass)}"
                if collision_mass is not None
                else "    $automass"
            ),
        ))
        if collision_concave:
            lines.append("    $concave")
        lines.append("}")
    for attachment in attachment_entries:
        lines.append(
            f"$attachment {_quoted(attachment.name)} "
            f"{_quoted(attachment.bone)} "
            f"{_qc_vector(attachment.position)} rotate "
            f"{_qc_vector(attachment.rotation)}"
        )
    for hitbox in hitbox_entries:
        lines.append(
            f"$hbox {hitbox.group} {_quoted(hitbox.bone)} "
            f"{_qc_vector(hitbox.minimum)} {_qc_vector(hitbox.maximum)}"
        )
    lines.extend((
        f"$sequence {_quoted('idle')} {_quoted(smd_name)} fps 1",
        "",
    ))
    return "\n".join(lines)


def serialize_vmt(
        material: AuthoringMaterial,
        base_texture: ContentRelativePath,
        *,
        normal_texture: ContentRelativePath | None = None,
) -> str:
    lines = [
        _quoted("VertexLitGeneric"),
        "{",
        f"    {_quoted('$basetexture')} {_quoted(_texture_reference(base_texture))}",
    ]
    if normal_texture is not None:
        lines.append(
            f"    {_quoted('$bumpmap')} {_quoted(_texture_reference(normal_texture))}"
        )
    tint = material.base_color[:3]
    if tint != (1.0, 1.0, 1.0):
        lines.append(
            f"    {_quoted('$color2')} {_quoted('[' + ' '.join(_float(value) for value in tint) + ']')}"
        )
    alpha = material.base_color[3]
    if alpha != 1.0:
        lines.append(f"    {_quoted('$alpha')} {_quoted(_float(alpha))}")
    if material.alpha_mode == MaterialAlphaMode.MASK:
        lines.extend((
            f"    {_quoted('$alphatest')} {_quoted('1')}",
            f"    {_quoted('$alphatestreference')} {_quoted(_float(material.alpha_cutoff))}",
        ))
    elif material.alpha_mode == MaterialAlphaMode.BLEND:
        lines.append(f"    {_quoted('$translucent')} {_quoted('1')}")
    if material.double_sided:
        lines.append(f"    {_quoted('$nocull')} {_quoted('1')}")
    if (
            isinstance(material, Source1AuthoringMaterial)
            and material.phong is not None
    ):
        lines.append(f"    {_quoted('$phong')} {_quoted('1')}")
        if material.phong.exponent is not None:
            lines.append(
                f"    {_quoted('$phongexponent')} "
                f"{_quoted(_float(material.phong.exponent))}"
            )
        if material.phong.boost is not None:
            lines.append(
                f"    {_quoted('$phongboost')} "
                f"{_quoted(_float(material.phong.boost))}"
            )
        if material.phong.tint is not None:
            lines.append(
                f"    {_quoted('$phongtint')} "
                f"{_quoted('[' + ' '.join(_float(value) for value in material.phong.tint) + ']')}"
            )
        if material.phong.fresnel_ranges is not None:
            lines.append(
                f"    {_quoted('$phongfresnelranges')} "
                f"{_quoted('[' + ' '.join(_float(value) for value in material.phong.fresnel_ranges) + ']')}"
            )
        if material.phong.albedo_tint:
            lines.append(
                f"    {_quoted('$phongalbedotint')} {_quoted('1')}"
            )
        if material.phong.rim_light:
            lines.append(f"    {_quoted('$rimlight')} {_quoted('1')}")
            if material.phong.rim_exponent is not None:
                lines.append(
                    f"    {_quoted('$rimlightexponent')} "
                    f"{_quoted(_float(material.phong.rim_exponent))}"
                )
            if material.phong.rim_boost is not None:
                lines.append(
                    f"    {_quoted('$rimlightboost')} "
                    f"{_quoted(_float(material.phong.rim_boost))}"
                )
    if (
            isinstance(material, Source1AuthoringMaterial)
            and material.parameters is not None
    ):
        parameters = material.parameters
        if parameters.self_illum:
            lines.append(f"    {_quoted('$selfillum')} {_quoted('1')}")
            if parameters.self_illum_tint is not None:
                lines.append(
                    f"    {_quoted('$selfillumtint')} "
                    f"{_quoted('[' + ' '.join(_float(value) for value in parameters.self_illum_tint) + ']')}"
                )
        if parameters.envmap is not None:
            lines.append(
                f"    {_quoted('$envmap')} {_quoted(parameters.envmap)}"
            )
            if parameters.envmap_tint is not None:
                lines.append(
                    f"    {_quoted('$envmaptint')} "
                    f"{_quoted('[' + ' '.join(_float(value) for value in parameters.envmap_tint) + ']')}"
                )
            if parameters.envmap_contrast is not None:
                lines.append(
                    f"    {_quoted('$envmapcontrast')} "
                    f"{_quoted(_float(parameters.envmap_contrast))}"
                )
            if parameters.envmap_saturation is not None:
                lines.append(
                    f"    {_quoted('$envmapsaturation')} "
                    f"{_quoted(_float(parameters.envmap_saturation))}"
                )
    lines.extend(("}", ""))
    return "\n".join(lines)


def serialize_vmt_patch(material_path: ContentRelativePath) -> str:
    path = ContentRelativePath(material_path)
    if (
            path.suffix.casefold() != ".vmt"
            or path.parts[0].casefold() == "materials"
    ):
        raise ValueError(
            "Source 1 material patches require material-relative .vmt paths"
        )
    return "\n".join((
        _quoted("Patch"),
        "{",
        f"    {_quoted('include')} {_quoted(path.as_posix())}",
        "}",
        "",
    ))


def serialize_vtex_settings(
        role: TextureRole,
        *,
        alpha_mode: MaterialAlphaMode = MaterialAlphaMode.OPAQUE,
) -> str:
    if role == TextureRole.BASE_COLOR:
        lines = [
            "numchannels 4"
            if alpha_mode != MaterialAlphaMode.OPAQUE
            else "numchannels 3"
        ]
        if alpha_mode != MaterialAlphaMode.OPAQUE:
            lines.append("dxt5 1")
    elif role == TextureRole.NORMAL:
        lines = [
            "normal 1",
            "numchannels 3",
        ]
    else:
        raise ValueError(f"Source 1 VTEX settings do not support {role.value} textures")
    return "\n".join((*lines, ""))
