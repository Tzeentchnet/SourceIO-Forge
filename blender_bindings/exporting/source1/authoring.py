from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from mathutils import Matrix, Vector

from ....library.source1.authoring import (
    Source1Attachment,
    Source1AuthoringBackend,
    Source1Bone,
    Source1Bodygroup,
    Source1BodygroupChoice,
    Source1AuthoringCollision,
    Source1AuthoringExecutionResult,
    Source1AuthoringLod,
    Source1AuthoringMaterial,
    Source1AuthoringPlan,
    Source1AuthoringRequest,
    Source1AuthoringService,
    Source1AuthoringTarget,
    Source1ExtractedAuthoringModel,
    Source1Hitbox,
    Source1MaterialParameters,
    Source1PhongParameters,
    Source1Skeleton,
    Source1SkinFamily,
    source1_pbr_to_phong,
)
from ....library.source2.authoring import (
    AuthoringGrouping,
    AuthoringMaterial,
    AuthoringScope,
    TextureRole,
    suggest_asset_name,
)
from ....library.source2.export import (
    DiagnosticSeverity,
    LossKind,
    LossReport,
)
from ....library.source2.workshop import ContentRelativePath
from ....library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from ..source2.authoring import (
    _export_group,
    _extract_materials,
    _extract_principled_material,
    _preserve_context,
    extract_blender_authoring_models,
    resolve_authoring_objects,
)


_PHONG_PROPERTIES = {
    "exponent": "sourceio_source1_phong_exponent",
    "boost": "sourceio_source1_phong_boost",
    "tint": "sourceio_source1_phong_tint",
    "fresnel_ranges": "sourceio_source1_phong_fresnel_ranges",
    "rim_exponent": "sourceio_source1_rimlight_exponent",
    "rim_boost": "sourceio_source1_rimlight_boost",
}
_COLLECTION_ROLE = "sourceio_source1_role"
_COLLECTION_LOD_DISTANCE = "sourceio_source1_lod_distance"
_COLLECTION_COLLISION_MASS = "sourceio_source1_collision_mass"
_COLLECTION_COLLISION_CONCAVE = "sourceio_source1_collision_concave"
_COLLECTION_BODYGROUP_NAME = "sourceio_source1_bodygroup_name"
_COLLECTION_BODYGROUP_CHOICE = "sourceio_source1_bodygroup_choice"
_COLLECTION_BODYGROUP_BLANK = "sourceio_source1_bodygroup_blank"
_COLLECTION_ROLES = frozenset((
    "NONE",
    "REFERENCE",
    "COLLISION",
    "LOD",
    "BODYGROUP",
))
_BLENDER_TO_SOURCE = Matrix((
    (0.0, 1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def _source1_material_path(
        material,
        report: LossReport,
) -> ContentRelativePath | None:
    explicit = (
        material.get("sourceio_export_source1_material_path")
        or material.get("sourceio_export_material_path")
    )
    value = explicit or material.get("full_path") or material.get("source_path")
    if not value:
        return None
    raw = str(value).split("?", 1)[0].replace("\\", "/").lstrip("/")
    if raw.casefold().startswith("materials/"):
        raw = raw[len("materials/"):]
    suffix = Path(raw).suffix.casefold()
    if not suffix:
        raw += ".vmt"
    try:
        path = ContentRelativePath(raw)
    except ValueError:
        path = None
    if path is None or path.suffix.casefold() != ".vmt":
        if explicit is not None:
            report.record(
                "authoring.source1.material.override.invalid",
                "The explicit SourceIO Source 1 material path must be a safe, "
                "material-relative editable .vmt path.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                path=("materials", material.name),
                details={"value": str(explicit)},
            )
        return None
    return path


def _phong_parameters(
        material,
        extracted: AuthoringMaterial,
        report: LossReport,
) -> tuple[Source1PhongParameters | None, bool]:
    enabled = bool(material.get("sourceio_source1_phong", False))
    convert_pbr = bool(material.get(
        "sourceio_source1_pbr_to_phong",
        False,
    ))
    values = {
        field_name: material.get(property_name)
        for field_name, property_name in _PHONG_PROPERTIES.items()
    }
    values["albedo_tint"] = bool(material.get(
        "sourceio_source1_phong_albedo_tint",
        False,
    ))
    values["rim_light"] = bool(material.get(
        "sourceio_source1_rimlight",
        False,
    ))
    configured = any(
        property_name in material
        for property_name in (
            *_PHONG_PROPERTIES.values(),
            "sourceio_source1_phong_albedo_tint",
            "sourceio_source1_rimlight",
        )
    )
    if enabled and convert_pbr:
        report.record(
            "authoring.source1.material.phong_mode_conflict",
            "Explicit phong parameters and PBR-to-phong conversion are "
            "mutually exclusive.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.SAFETY,
            path=("materials", material.name),
        )
        return None, False
    if convert_pbr:
        pbr_textures = sorted(
            texture.role.value for texture in extracted.textures
            if texture.role in {TextureRole.ROUGHNESS, TextureRole.METALLIC}
        )
        if pbr_textures:
            report.record(
                "authoring.source1.material.pbr_texture_conversion_deferred",
                "The validated PBR-to-phong path currently accepts scalar "
                "roughness and metallic values only.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.DEFERRED,
                path=("materials", material.name),
                details={"roles": pbr_textures},
            )
            return None, False
        try:
            return source1_pbr_to_phong(
                extracted.roughness,
                extracted.metallic,
                max_exponent=material.get(
                    "sourceio_source1_pbr_max_exponent",
                    150.0,
                ),
                dielectric_boost=material.get(
                    "sourceio_source1_pbr_dielectric_boost",
                    0.25,
                ),
            ), True
        except (TypeError, ValueError) as exc:
            report.record(
                "authoring.source1.material.pbr_conversion_invalid",
                str(exc),
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                path=("materials", material.name),
            )
            return None, False
    if not enabled:
        if configured:
            report.record(
                "authoring.source1.material.phong.disabled",
                "Explicit Source 1 phong values require "
                "sourceio_source1_phong to be enabled.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                path=("materials", material.name),
            )
        return None, False
    try:
        return Source1PhongParameters(**values), False
    except (TypeError, ValueError) as exc:
        report.record(
            "authoring.source1.material.phong.invalid",
            str(exc),
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.SAFETY,
            path=("materials", material.name),
        )
        return None, False


def _material_parameters(
        material,
        report: LossReport,
) -> Source1MaterialParameters | None:
    values = {
        "self_illum": bool(material.get(
            "sourceio_source1_selfillum",
            False,
        )),
        "self_illum_tint": material.get(
            "sourceio_source1_selfillum_tint"
        ),
        "envmap": material.get("sourceio_source1_envmap"),
        "envmap_tint": material.get("sourceio_source1_envmap_tint"),
        "envmap_contrast": material.get(
            "sourceio_source1_envmap_contrast"
        ),
        "envmap_saturation": material.get(
            "sourceio_source1_envmap_saturation"
        ),
    }
    if not any(
            property_name in material
            for property_name in (
                "sourceio_source1_selfillum",
                "sourceio_source1_selfillum_tint",
                "sourceio_source1_envmap",
                "sourceio_source1_envmap_tint",
                "sourceio_source1_envmap_contrast",
                "sourceio_source1_envmap_saturation",
            )
    ):
        return None
    try:
        return Source1MaterialParameters(**values)
    except (TypeError, ValueError) as exc:
        report.record(
            "authoring.source1.material.parameters_invalid",
            str(exc),
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.SAFETY,
            path=("materials", material.name),
        )
        return None


def _extract_source1_material(
        material,
        name: str,
        report: LossReport,
) -> AuthoringMaterial | None:
    resource_path = _source1_material_path(material, report)
    if resource_path is not None:
        return Source1AuthoringMaterial(name=name, existing_vmt=resource_path)
    extracted = _extract_principled_material(material, name, report)
    if extracted is None:
        return None
    phong, pbr_converted = _phong_parameters(material, extracted, report)
    parameters = _material_parameters(material, report)
    return Source1AuthoringMaterial(
        name=extracted.name,
        base_color=extracted.base_color,
        metallic=extracted.metallic,
        roughness=extracted.roughness,
        alpha_mode=extracted.alpha_mode,
        alpha_cutoff=extracted.alpha_cutoff,
        double_sided=extracted.double_sided,
        textures=extracted.textures,
        phong=phong,
        parameters=parameters,
        pbr_converted=pbr_converted,
    )


def _extract_skin_families(
        objects: tuple,
        materials: tuple,
        material_names: dict[int, str],
        report: LossReport,
) -> tuple[Source1SkinFamily, ...]:
    source_by_name = {}
    for obj in objects:
        for material in obj.data.materials:
            if material is None:
                continue
            authored_name = material_names.get(material.as_pointer())
            if authored_name is not None:
                source_by_name[authored_name] = material
    replacements: dict[str, dict[str, str]] = {}
    for material in materials:
        source = source_by_name.get(material.name)
        values = (
            source.get("sourceio_source1_skins")
            if source is not None
            else None
        )
        if values is None:
            continue
        if not hasattr(values, "items"):
            report.record(
                "authoring.source1.skin.mapping_invalid",
                "Source 1 skin replacements must be a mapping from family "
                "names to bare VMT names.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                path=("materials", material.name),
            )
            continue
        for family_name, replacement in values.items():
            family_name = str(family_name).strip()
            replacement = str(replacement).strip()
            if not family_name:
                report.record(
                    "authoring.source1.skin.name_invalid",
                    "Source 1 skin family names cannot be empty.",
                    severity=DiagnosticSeverity.ERROR,
                    loss_kind=LossKind.SAFETY,
                    path=("materials", material.name),
                )
                continue
            replacements.setdefault(family_name, {})[
                material.name
            ] = replacement
    if not replacements:
        return ()
    base_names = tuple(material.name for material in materials)
    families = [Source1SkinFamily(base_names)]
    for family_name, remaps in sorted(
            replacements.items(),
            key=lambda item: (item[0].casefold(), item[0]),
    ):
        try:
            families.append(Source1SkinFamily(tuple(
                remaps.get(material_name, material_name)
                for material_name in base_names
            )))
        except (TypeError, ValueError) as exc:
            report.record(
                "authoring.source1.skin.family_invalid",
                str(exc),
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={"family": family_name},
            )
    return tuple(families)


def _collection_setting(collection, name: str, default):
    stored = collection.get(name)
    return getattr(collection, name, default) if stored is None else stored


def _walk_collections(root) -> tuple:
    if root is None:
        return ()
    result = []
    seen = set()
    pending = [root]
    while pending:
        collection = pending.pop()
        pointer = collection.as_pointer()
        if pointer in seen:
            continue
        seen.add(pointer)
        result.append(collection)
        pending.extend(sorted(
            collection.children,
            key=lambda item: (item.name.casefold(), item.name),
            reverse=True,
        ))
    return tuple(result)


def _collections_in_scope(context, scope: AuthoringScope, object_pointers: set[int]):
    if scope == AuthoringScope.COLLECTION:
        candidates = _walk_collections(context.collection)
    else:
        candidates = _walk_collections(context.scene.collection)
    if scope == AuthoringScope.SELECTED:
        candidates = tuple(
            collection
            for collection in candidates
            if any(
                obj.type == "MESH" and obj.as_pointer() in object_pointers
                for obj in collection.all_objects
            )
        )
    return tuple(sorted(
        candidates,
        key=lambda item: (item.name.casefold(), item.name),
    ))


def _resolve_source1_collection_roles(
        context,
        request: Source1AuthoringRequest,
        objects: tuple,
        report: LossReport,
):
    object_by_pointer = {obj.as_pointer(): obj for obj in objects}
    tagged = []
    for collection in _collections_in_scope(
            context,
            request.scope,
            set(object_by_pointer),
    ):
        role = str(_collection_setting(collection, _COLLECTION_ROLE, "NONE")).upper()
        if role == "NONE":
            continue
        if role not in _COLLECTION_ROLES:
            report.record(
                "authoring.source1.collection.role_invalid",
                "Source 1 collection roles must be reference, collision, LOD, "
                "bodygroup, or unassigned.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={
                    "collection": collection.name,
                    "role": role,
                },
            )
            continue
        members = tuple(sorted(
            (
                object_by_pointer[obj.as_pointer()]
                for obj in collection.all_objects
                if obj.type == "MESH" and obj.as_pointer() in object_by_pointer
            ),
            key=lambda item: (item.name.casefold(), item.name),
        ))
        is_blank_bodygroup = (
            role == "BODYGROUP"
            and bool(_collection_setting(
                collection,
                _COLLECTION_BODYGROUP_BLANK,
                False,
            ))
        )
        if not members and not is_blank_bodygroup:
            report.record(
                "authoring.source1.collection.empty",
                "A tagged Source 1 authoring collection contains no mesh "
                "objects in the selected export scope.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.MISSING,
                details={
                    "collection": collection.name,
                    "role": role.casefold(),
                },
            )
        tagged.append((collection, role, members))

    if not tagged:
        return objects, None, (), (), False

    memberships: dict[int, list[str]] = {}
    for collection, role, members in tagged:
        for obj in members:
            memberships.setdefault(obj.as_pointer(), []).append(
                f"{role.casefold()}:{collection.name}"
            )
    for pointer, assignments in memberships.items():
        if len(assignments) > 1:
            report.record(
                "authoring.source1.collection.overlap",
                "A mesh object cannot belong to more than one tagged Source 1 "
                "authoring collection.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={
                    "object": object_by_pointer[pointer].name,
                    "assignments": assignments,
                },
            )

    reference_records = tuple(item for item in tagged if item[1] == "REFERENCE")
    collision_records = tuple(item for item in tagged if item[1] == "COLLISION")
    lod_records = tuple(item for item in tagged if item[1] == "LOD")
    bodygroup_records = tuple(
        item for item in tagged if item[1] == "BODYGROUP"
    )
    if len(collision_records) > 1:
        report.record(
            "authoring.source1.collision.multiple",
            "A Source 1 asset can use only one tagged collision collection.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.SAFETY,
            details={
                "collections": [item[0].name for item in collision_records],
            },
        )

    special_pointers = {
        obj.as_pointer()
        for _, role, members in tagged
        if role in {"COLLISION", "LOD", "BODYGROUP"}
        for obj in members
    }
    if reference_records:
        reference_pointers = {
            obj.as_pointer()
            for _, _, members in reference_records
            for obj in members
        }
    else:
        reference_pointers = set(object_by_pointer) - special_pointers
    reference_objects = tuple(sorted(
        (object_by_pointer[pointer] for pointer in reference_pointers),
        key=lambda item: (item.name.casefold(), item.name),
    ))
    if not reference_objects:
        report.record(
            "authoring.source1.reference.empty",
            "Tagged Source 1 authoring requires reference mesh geometry.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.MISSING,
        )

    collision = None
    if collision_records:
        collection, _, members = collision_records[0]
        mass_value = float(_collection_setting(
            collection,
            _COLLECTION_COLLISION_MASS,
            0.0,
        ))
        if not math.isfinite(mass_value) or mass_value < 0:
            report.record(
                "authoring.source1.collision.mass_invalid",
                "Source 1 collision mass must be zero for automatic mass or "
                "a finite positive value.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={
                    "collection": collection.name,
                    "mass": mass_value,
                },
            )
        collision = (
            members,
            None if mass_value == 0 else mass_value,
            bool(_collection_setting(
                collection,
                _COLLECTION_COLLISION_CONCAVE,
                True,
            )),
        )

    lods = []
    seen_distances: dict[float, str] = {}
    for collection, _, members in lod_records:
        distance = float(_collection_setting(
            collection,
            _COLLECTION_LOD_DISTANCE,
            10.0,
        ))
        if not math.isfinite(distance) or distance <= 0:
            report.record(
                "authoring.source1.lod.distance_invalid",
                "Source 1 LOD distances must be finite and positive.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={
                    "collection": collection.name,
                    "distance": distance,
                },
            )
            continue
        previous = seen_distances.get(distance)
        if previous is not None:
            report.record(
                "authoring.source1.lod.distance_duplicate",
                "Source 1 LOD switch distances must be unique.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={
                    "first": previous,
                    "second": collection.name,
                    "distance": distance,
                },
            )
            continue
        seen_distances[distance] = collection.name
        lods.append((distance, members))

    bodygroups: dict[str, list[tuple[str, tuple | None]]] = {}
    seen_bodygroup_choices: set[tuple[str, str]] = set()
    for collection, _, members in bodygroup_records:
        name = str(_collection_setting(
            collection,
            _COLLECTION_BODYGROUP_NAME,
            "",
        )).strip()
        choice = str(_collection_setting(
            collection,
            _COLLECTION_BODYGROUP_CHOICE,
            "",
        )).strip()
        blank = bool(_collection_setting(
            collection,
            _COLLECTION_BODYGROUP_BLANK,
            False,
        ))
        if not name or any(ord(character) < 32 for character in name):
            report.record(
                "authoring.source1.bodygroup.name_invalid",
                "A Source 1 bodygroup collection requires a non-empty name "
                "without control characters.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={"collection": collection.name},
            )
            continue
        if (
                not choice
                or any(ord(character) < 32 for character in choice)
                or not all(
                    character.isascii()
                    and (character.islower() or character.isdigit() or character == "_")
                    for character in choice
                )
        ):
            report.record(
                "authoring.source1.bodygroup.choice_invalid",
                "Source 1 bodygroup choice names must contain only lowercase "
                "ASCII letters, digits, and underscores.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={"collection": collection.name, "choice": choice},
            )
            continue
        key = (name.casefold(), choice.casefold())
        if key in seen_bodygroup_choices:
            report.record(
                "authoring.source1.bodygroup.choice_duplicate",
                "Source 1 bodygroup choice names must be unique within their "
                "bodygroup.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={"bodygroup": name, "choice": choice},
            )
            continue
        seen_bodygroup_choices.add(key)
        bodygroups.setdefault(name, []).append(
            (choice, None if blank else members)
        )

    return (
        reference_objects,
        collision,
        tuple(sorted(lods, key=lambda item: item[0])),
        tuple(
            (
                name,
                tuple(sorted(
                    choices,
                    key=lambda item: (item[0].casefold(), item[0]),
                )),
            )
            for name, choices in sorted(
                bodygroups.items(),
                key=lambda item: (item[0].casefold(), item[0]),
            )
        ),
        True,
    )


def _source_matrix(matrix) -> Matrix:
    return _BLENDER_TO_SOURCE @ Matrix(matrix) @ _BLENDER_TO_SOURCE.inverted()


def _find_source1_armature(objects: tuple, report: LossReport):
    armatures = {
        armature.as_pointer(): armature
        for obj in objects
        if (armature := obj.find_armature()) is not None
    }
    if not armatures:
        report.record(
            "authoring.source1.skeleton.missing",
            "Skeletal Source 1 export requires an Armature modifier or "
            "armature parent on the reference meshes.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.MISSING,
        )
        return None
    if len(armatures) > 1:
        report.record(
            "authoring.source1.skeleton.multiple",
            "A Source 1 model can use only one armature.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.SAFETY,
            details={
                "armatures": sorted(
                    armature.name for armature in armatures.values()
                ),
            },
        )
        return None
    return next(iter(armatures.values()))


def _source1_skeleton(armature) -> Source1Skeleton:
    ordered = []

    def append_bone(bone):
        ordered.append(bone)
        for child in sorted(
                bone.children,
                key=lambda item: (item.name.casefold(), item.name),
        ):
            append_bone(child)

    for root in sorted(
            (bone for bone in armature.data.bones if bone.parent is None),
            key=lambda item: (item.name.casefold(), item.name),
    ):
        append_bone(root)
    indices = {bone.name: index for index, bone in enumerate(ordered)}
    world_matrices = {}
    bones = []
    for bone in ordered:
        world = _source_matrix(armature.matrix_world @ bone.matrix_local)
        local = (
            world_matrices[bone.parent.name].inverted_safe() @ world
            if bone.parent is not None
            else world
        )
        _, _, scale = local.decompose()
        if any(abs(float(component) - 1.0) > 1e-5 for component in scale):
            raise ValueError(
                f"Source 1 bone {bone.name!r} has unsupported rest-pose scale"
            )
        position = tuple(
            float(component) / SOURCE1_HAMMER_UNIT_TO_METERS
            for component in local.to_translation()
        )
        rotation = tuple(float(component) for component in local.to_euler("XYZ"))
        bones.append(Source1Bone(
            bone.name,
            indices[bone.parent.name] if bone.parent is not None else -1,
            position,
            rotation,
        ))
        world_matrices[bone.name] = world
    return Source1Skeleton(tuple(bones))


@contextmanager
def _rest_pose(armature):
    if armature is None:
        yield
        return
    previous = armature.data.pose_position
    armature.data.pose_position = "REST"
    try:
        armature.update_tag()
        yield
    finally:
        armature.data.pose_position = previous
        armature.update_tag()


def _apply_skeleton_weights(
        document,
        objects: tuple,
        skeleton: Source1Skeleton,
        report: LossReport,
):
    sources = {obj.name: obj for obj in objects}
    bone_indices = {
        bone.name: index for index, bone in enumerate(skeleton.bones)
    }
    meshes = []
    for mesh in document.meshes:
        source_name = str(mesh.metadata.get("blender_object", mesh.name))
        source = sources.get(source_name)
        if source is None:
            report.record(
                "authoring.source1.skeleton.mesh_source_missing",
                "An evaluated mesh cannot be matched to its source object.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.MISSING,
                details={"mesh": mesh.name, "source": source_name},
            )
            meshes.append(mesh)
            continue
        group_to_bone = {
            group.index: bone_indices[group.name]
            for group in source.vertex_groups
            if group.name in bone_indices
        }
        vertices = []
        missing = 0
        for vertex in mesh.vertices:
            combined = {}
            for group, weight in zip(
                    vertex.corner.bone_indices,
                    vertex.corner.bone_weights,
            ):
                index = group_to_bone.get(int(group))
                weight = float(weight)
                if index is None or not math.isfinite(weight) or weight <= 0:
                    continue
                combined[index] = combined.get(index, 0.0) + weight
            if not combined:
                missing += 1
                vertices.append(vertex)
                continue
            ordered = tuple(sorted(
                combined.items(),
                key=lambda item: (-item[1], item[0]),
            ))
            vertices.append(replace(
                vertex,
                corner=replace(
                    vertex.corner,
                    bone_indices=tuple(index for index, _ in ordered),
                    bone_weights=tuple(weight for _, weight in ordered),
                ),
            ))
        if missing:
            report.record(
                "authoring.source1.skeleton.weights_missing",
                "Every skeletal Source 1 vertex must have a positive weight "
                "for a bone in the exported armature.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.MISSING,
                details={"mesh": mesh.name, "vertices": missing},
            )
        meshes.append(replace(mesh, vertices=tuple(vertices)))
    return replace(document, meshes=tuple(meshes))


def _object_setting(obj, name: str, default):
    stored = obj.get(name)
    return getattr(obj, name, default) if stored is None else stored


def _object_bone(obj, armature, default: str) -> str:
    if (
            obj.parent is armature
            and obj.parent_type == "BONE"
            and obj.parent_bone
    ):
        return str(obj.parent_bone)
    value = str(_object_setting(
        obj,
        "sourceio_source1_bone",
        default,
    )).strip()
    return value or default


def _bone_source_matrix(armature, bone_name: str) -> Matrix:
    if armature is None and bone_name == "root":
        return Matrix.Identity(4)
    bone = armature.data.bones.get(bone_name) if armature is not None else None
    if bone is None:
        raise ValueError(f"Source 1 authoring bone {bone_name!r} is unavailable")
    return _source_matrix(armature.matrix_world @ bone.matrix_local)


def _extract_attachments_and_hitboxes(
        context,
        armature,
        skeleton: Source1Skeleton | None,
        report: LossReport,
) -> tuple[tuple[Source1Attachment, ...], tuple[Source1Hitbox, ...]]:
    default_bone = skeleton.bones[0].name if skeleton is not None else "root"
    attachments = []
    hitboxes = []
    for obj in sorted(
            context.scene.objects,
            key=lambda item: (item.name.casefold(), item.name),
    ):
        is_attachment = bool(_object_setting(
            obj,
            "sourceio_source1_attachment",
            False,
        ))
        is_hitbox = bool(_object_setting(
            obj,
            "sourceio_source1_hitbox",
            False,
        ))
        if not is_attachment and not is_hitbox:
            continue
        if is_attachment and is_hitbox:
            report.record(
                "authoring.source1.helper.role_conflict",
                "A Blender object cannot be both a Source 1 attachment and "
                "hitbox.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={"object": obj.name},
            )
            continue
        bone_name = _object_bone(obj, armature, default_bone)
        try:
            bone_world = _bone_source_matrix(armature, bone_name)
        except ValueError as exc:
            report.record(
                "authoring.source1.helper.bone_missing",
                str(exc),
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.MISSING,
                details={"object": obj.name},
            )
            continue
        object_world = _source_matrix(obj.matrix_world)
        relative = bone_world.inverted_safe() @ object_world
        if is_attachment:
            position = tuple(
                float(component) / SOURCE1_HAMMER_UNIT_TO_METERS
                for component in relative.to_translation()
            )
            rotation = tuple(
                math.degrees(float(component))
                for component in relative.to_euler("XYZ")
            )
            try:
                attachments.append(Source1Attachment(
                    obj.name,
                    bone_name,
                    position,
                    rotation,
                ))
            except (TypeError, ValueError) as exc:
                report.record(
                    "authoring.source1.attachment.invalid",
                    str(exc),
                    severity=DiagnosticSeverity.ERROR,
                    loss_kind=LossKind.SAFETY,
                    details={"object": obj.name},
                )
            continue
        if obj.type == "EMPTY":
            radius = float(obj.empty_display_size)
            corners = tuple(
                Vector((x * radius, y * radius, z * radius))
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            )
        else:
            corners = tuple(Vector(corner) for corner in obj.bound_box)
        points = tuple(
            relative @ corner
            for corner in corners
        )
        minimum = tuple(
            min(float(point[axis]) for point in points)
            / SOURCE1_HAMMER_UNIT_TO_METERS
            for axis in range(3)
        )
        maximum = tuple(
            max(float(point[axis]) for point in points)
            / SOURCE1_HAMMER_UNIT_TO_METERS
            for axis in range(3)
        )
        try:
            hitboxes.append(Source1Hitbox(
                int(_object_setting(
                    obj,
                    "sourceio_source1_hitbox_group",
                    0,
                )),
                bone_name,
                minimum,
                maximum,
            ))
        except (TypeError, ValueError) as exc:
            report.record(
                "authoring.source1.hitbox.invalid",
                str(exc),
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                details={"object": obj.name},
            )
    return tuple(attachments), tuple(hitboxes)


def extract_source1_authoring_models(
        context,
        request: Source1AuthoringRequest,
        *,
        report: LossReport | None = None,
):
    loss_report = report if report is not None else LossReport()
    objects = tuple(
        obj for obj in resolve_authoring_objects(context, request.scope)
        if not bool(_object_setting(
            obj,
            "sourceio_source1_hitbox",
            False,
        ))
    )
    if not objects:
        return extract_blender_authoring_models(
            context,
            request,
            material_extractor=_extract_source1_material,
            unit_scale=SOURCE1_HAMMER_UNIT_TO_METERS,
            fbx_exporter=None,
            report=loss_report,
        )
    reference_objects, collision, lods, bodygroups, _ = (
        _resolve_source1_collection_roles(
            context,
            request,
            objects,
            loss_report,
        )
    )
    if (
            request.grouping == AuthoringGrouping.BATCH
            and (collision is not None or lods or bodygroups)
    ):
        loss_report.record(
            "authoring.source1.collection.batch_unsupported",
            "Tagged Source 1 collision, LOD, and bodygroup collections require "
            "combined model grouping.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.DEFERRED,
        )
    armature = None
    skeleton = None
    if request.export_skeleton:
        if request.grouping != AuthoringGrouping.COMBINED:
            loss_report.record(
                "authoring.source1.skeleton.batch_unsupported",
                "Skeletal Source 1 export requires combined model grouping.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.DEFERRED,
            )
        armature = _find_source1_armature(reference_objects, loss_report)
        if armature is not None:
            try:
                skeleton = _source1_skeleton(armature)
            except (TypeError, ValueError) as exc:
                loss_report.record(
                    "authoring.source1.skeleton.invalid",
                    str(exc),
                    severity=DiagnosticSeverity.ERROR,
                    loss_kind=LossKind.SAFETY,
                )
    if loss_report.has_errors:
        return (), loss_report

    material_objects = {
        obj.as_pointer(): obj
        for obj in reference_objects
    }
    for _, lod_objects in lods:
        for obj in lod_objects:
            material_objects[obj.as_pointer()] = obj
    for _, choices in bodygroups:
        for _, bodygroup_objects in choices:
            for obj in bodygroup_objects or ():
                material_objects[obj.as_pointer()] = obj
    material_scope = tuple(sorted(
        material_objects.values(),
        key=lambda item: (item.name.casefold(), item.name),
    ))
    materials, material_names = _extract_materials(
        material_scope,
        loss_report,
        _extract_source1_material,
    )
    if loss_report.has_errors:
        return (), loss_report
    skin_families = _extract_skin_families(
        material_scope,
        materials,
        material_names,
        loss_report,
    )
    attachments, hitboxes = _extract_attachments_and_hitboxes(
        context,
        armature,
        skeleton,
        loss_report,
    )
    if loss_report.has_errors:
        return (), loss_report

    if request.grouping == AuthoringGrouping.COMBINED:
        groups = ((request.asset_name, reference_objects),)
    else:
        groups = tuple(
            (suggest_asset_name(obj.name), (obj,))
            for obj in reference_objects
        )
        names = [name.casefold() for name, _ in groups]
        if len(names) != len(set(names)):
            loss_report.record(
                "authoring.model.name.collision",
                "Batch object names collide after path-safe normalization.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
            )
            return (), loss_report

    models = []
    with _preserve_context(context), _rest_pose(armature):
        context.view_layer.update()
        for name, group in groups:
            extracted = _export_group(
                context,
                group,
                name,
                materials,
                material_names,
                loss_report,
                None,
                SOURCE1_HAMMER_UNIT_TO_METERS,
            )
            if skeleton is not None:
                extracted = replace(
                    extracted,
                    document=_apply_skeleton_weights(
                        extracted.document,
                        group,
                        skeleton,
                        loss_report,
                    ),
                )
            collision_spec = None
            lod_specs = ()
            bodygroup_specs = ()
            if request.grouping == AuthoringGrouping.COMBINED:
                if collision is not None:
                    collision_objects, mass, concave = collision
                    collision_model = _export_group(
                        context,
                        collision_objects,
                        f"{name}_collision",
                        (),
                        material_names,
                        loss_report,
                        None,
                        SOURCE1_HAMMER_UNIT_TO_METERS,
                    )
                    collision_spec = Source1AuthoringCollision(
                        collision_model.document,
                        mass=mass,
                        concave=concave,
                    )
                lod_values = []
                for index, (distance, lod_objects) in enumerate(lods, start=1):
                    lod_model = _export_group(
                        context,
                        lod_objects,
                        f"{name}_lod{index}",
                        (),
                        material_names,
                        loss_report,
                        None,
                        SOURCE1_HAMMER_UNIT_TO_METERS,
                    )
                    lod_document = (
                        _apply_skeleton_weights(
                            lod_model.document,
                            lod_objects,
                            skeleton,
                            loss_report,
                        )
                        if skeleton is not None
                        else lod_model.document
                    )
                    lod_values.append(Source1AuthoringLod(
                        distance,
                        lod_document,
                    ))
                lod_specs = tuple(lod_values)
                bodygroup_values = []
                for bodygroup_name, choices in bodygroups:
                    choice_values = []
                    for choice_name, bodygroup_objects in choices:
                        if bodygroup_objects is None:
                            choice_values.append(Source1BodygroupChoice(
                                choice_name,
                                None,
                            ))
                            continue
                        bodygroup_model = _export_group(
                            context,
                            bodygroup_objects,
                            f"{name}_{choice_name}",
                            (),
                            material_names,
                            loss_report,
                            None,
                            SOURCE1_HAMMER_UNIT_TO_METERS,
                        )
                        bodygroup_document = (
                            _apply_skeleton_weights(
                                bodygroup_model.document,
                                bodygroup_objects,
                                skeleton,
                                loss_report,
                            )
                            if skeleton is not None
                            else bodygroup_model.document
                        )
                        choice_values.append(Source1BodygroupChoice(
                            choice_name,
                            bodygroup_document,
                        ))
                    bodygroup_values.append(Source1Bodygroup(
                        bodygroup_name,
                        tuple(choice_values),
                    ))
                bodygroup_specs = tuple(bodygroup_values)
            models.append(Source1ExtractedAuthoringModel(
                extracted.name,
                extracted.document,
                extracted.fbx_payload,
                extracted.materials,
                collision=collision_spec,
                lods=lod_specs,
                bodygroups=bodygroup_specs,
                skin_families=skin_families,
                attachments=attachments,
                hitboxes=hitboxes,
                skeleton=skeleton,
            ))
    return tuple(models), loss_report


@dataclass(frozen=True, slots=True)
class BlenderSource1AuthoringResult:
    plan: Source1AuthoringPlan
    execution: Source1AuthoringExecutionResult


def build_source1_authoring_plan(
        context,
        request: Source1AuthoringRequest,
) -> Source1AuthoringPlan:
    models, report = extract_source1_authoring_models(context, request)
    backend = Source1AuthoringBackend().build(request, models, report=report)
    return Source1AuthoringService().plan(request, backend)


def execute_source1_authoring(
        context,
        request: Source1AuthoringRequest,
        *,
        tool_root=None,
) -> BlenderSource1AuthoringResult:
    plan = build_source1_authoring_plan(context, request)
    execution = Source1AuthoringService().execute(
        plan,
        tool_root=tool_root,
    )
    return BlenderSource1AuthoringResult(plan, execution)


def recompile_source1_manifest(
        game_root,
        content_root,
        manifest_path: str | ContentRelativePath,
        *,
        tool_root=None,
):
    target = Source1AuthoringTarget(game_root, content_root)
    return Source1AuthoringService().recompile(
        target,
        manifest_path,
        tool_root=tool_root,
    )
