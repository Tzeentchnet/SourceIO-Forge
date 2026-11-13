from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ....library.source1.authoring import (
    Source1AuthoringBackend,
    Source1AuthoringCollision,
    Source1AuthoringExecutionResult,
    Source1AuthoringLod,
    Source1AuthoringMaterial,
    Source1AuthoringPlan,
    Source1AuthoringRequest,
    Source1AuthoringService,
    Source1AuthoringTarget,
    Source1ExtractedAuthoringModel,
    Source1PhongParameters,
)
from ....library.source2.authoring import (
    AuthoringGrouping,
    AuthoringMaterial,
    AuthoringScope,
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
}
_COLLECTION_ROLE = "sourceio_source1_role"
_COLLECTION_LOD_DISTANCE = "sourceio_source1_lod_distance"
_COLLECTION_COLLISION_MASS = "sourceio_source1_collision_mass"
_COLLECTION_COLLISION_CONCAVE = "sourceio_source1_collision_concave"
_COLLECTION_ROLES = frozenset(("NONE", "REFERENCE", "COLLISION", "LOD"))


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
        report: LossReport,
) -> Source1PhongParameters | None:
    enabled = bool(material.get("sourceio_source1_phong", False))
    values = {
        field_name: material.get(property_name)
        for field_name, property_name in _PHONG_PROPERTIES.items()
    }
    if not enabled:
        if any(value is not None for value in values.values()):
            report.record(
                "authoring.source1.material.phong.disabled",
                "Explicit Source 1 phong values require "
                "sourceio_source1_phong to be enabled.",
                severity=DiagnosticSeverity.ERROR,
                loss_kind=LossKind.SAFETY,
                path=("materials", material.name),
            )
        return None
    try:
        return Source1PhongParameters(**values)
    except (TypeError, ValueError) as exc:
        report.record(
            "authoring.source1.material.phong.invalid",
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
    phong = _phong_parameters(material, report)
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
    )


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
                "or unassigned.",
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
        if not members:
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
        return objects, None, (), False

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
        if role in {"COLLISION", "LOD"}
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

    return (
        reference_objects,
        collision,
        tuple(sorted(lods, key=lambda item: item[0])),
        True,
    )


def extract_source1_authoring_models(
        context,
        request: Source1AuthoringRequest,
        *,
        report: LossReport | None = None,
):
    loss_report = report if report is not None else LossReport()
    objects = resolve_authoring_objects(context, request.scope)
    if not objects:
        return extract_blender_authoring_models(
            context,
            request,
            material_extractor=_extract_source1_material,
            unit_scale=SOURCE1_HAMMER_UNIT_TO_METERS,
            fbx_exporter=None,
            report=loss_report,
        )
    reference_objects, collision, lods, has_tags = (
        _resolve_source1_collection_roles(
            context,
            request,
            objects,
            loss_report,
        )
    )
    if not has_tags:
        return extract_blender_authoring_models(
            context,
            request,
            material_extractor=_extract_source1_material,
            unit_scale=SOURCE1_HAMMER_UNIT_TO_METERS,
            fbx_exporter=None,
            report=loss_report,
        )
    if (
            request.grouping == AuthoringGrouping.BATCH
            and (collision is not None or lods)
    ):
        loss_report.record(
            "authoring.source1.collection.batch_unsupported",
            "Tagged Source 1 collision and LOD collections require combined "
            "model grouping.",
            severity=DiagnosticSeverity.ERROR,
            loss_kind=LossKind.DEFERRED,
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
    with _preserve_context(context):
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
            collision_spec = None
            lod_specs = ()
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
                    lod_values.append(Source1AuthoringLod(
                        distance,
                        lod_model.document,
                    ))
                lod_specs = tuple(lod_values)
            models.append(Source1ExtractedAuthoringModel(
                extracted.name,
                extracted.document,
                extracted.fbx_payload,
                extracted.materials,
                collision_spec,
                lod_specs,
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
