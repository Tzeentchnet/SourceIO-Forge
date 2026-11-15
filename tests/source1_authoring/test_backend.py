from __future__ import annotations

import json
from pathlib import Path

import pytest

from SourceIO.library.models.smd import parse as parse_smd
from SourceIO.library.source1.authoring import (
    Source1ArtifactKind,
    Source1ArtifactRoot,
    Source1Attachment,
    Source1AuthoringArtifact,
    Source1AuthoringBackend,
    Source1AuthoringCollision,
    Source1AuthoringLod,
    Source1AuthoringManifest,
    Source1AuthoringManifestError,
    Source1AuthoringMaterial,
    Source1AuthoringRequest,
    Source1AuthoringService,
    Source1AuthoringTarget,
    Source1AuthoringTargetError,
    Source1Bone,
    Source1Bodygroup,
    Source1BodygroupChoice,
    Source1ExtractedAuthoringModel,
    Source1Hitbox,
    Source1MaterialParameters,
    Source1PhongParameters,
    Source1Skeleton,
    Source1SkinFamily,
    serialize_qc,
    serialize_smd,
    serialize_vmt,
    serialize_vmt_patch,
    serialize_vtex_settings,
    source1_pbr_to_phong,
    stage_source1_artifacts,
)
from SourceIO.library.source2.authoring import (
    AuthoringMaterial,
    AuthoringTexture,
    ExtractedAuthoringModel,
    MaterialAlphaMode,
    TextureRole,
    solid_color_tga,
)
from SourceIO.library.source2.export import (
    CornerData,
    ExportBlockedError,
    MeshFace,
    MeshVertex,
    ModelDocument,
    StaticMesh,
)
from SourceIO.library.source2.workshop import ContentRelativePath


FIXTURES = Path(__file__).parents[1] / "fixtures" / "source1_authoring"


def _target(tmp_path: Path, name: str = "tf") -> Source1AuthoringTarget:
    game_root = tmp_path / name
    content_root = tmp_path / f"{name}_content"
    game_root.mkdir()
    content_root.mkdir()
    (game_root / "gameinfo.txt").write_text("GameInfo {}\n", encoding="utf-8")
    return Source1AuthoringTarget(game_root, content_root)


def _document(*, material: str = "crate", uv: bool = True) -> ModelDocument:
    texcoords = (
        ((0.0, 1.0),),
        ((1.0, 1.0),),
        ((0.0, 0.0),),
    )
    vertices = tuple(
        MeshVertex(
            position,
            CornerData(
                normal=(0.0, 0.0, 1.0),
                texcoords=texcoords[index] if uv else (),
            ),
        )
        for index, position in enumerate((
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ))
    )
    return ModelDocument(
        "crate",
        (StaticMesh(
            "crate_mesh",
            vertices,
            (MeshFace((0, 1, 2), material),),
        ),),
    )


def _material(
        *,
        metallic: float = 0.0,
        roughness: float = 0.5,
        alpha_mode: MaterialAlphaMode = MaterialAlphaMode.MASK,
) -> AuthoringMaterial:
    return AuthoringMaterial(
        "crate",
        base_color=(0.25, 0.5, 0.75, 1.0),
        metallic=metallic,
        roughness=roughness,
        alpha_mode=alpha_mode,
        alpha_cutoff=0.4,
        double_sided=True,
        textures=(
            AuthoringTexture(
                "crate_base_color",
                TextureRole.BASE_COLOR,
                solid_color_tga((0.2, 0.4, 0.6, 0.5)),
                extension=".tga",
                srgb=True,
            ),
            AuthoringTexture(
                "crate_alpha",
                TextureRole.ALPHA,
                solid_color_tga((0.5, 0.5, 0.5, 1.0)),
                extension=".tga",
            ),
            AuthoringTexture(
                "crate_normal",
                TextureRole.NORMAL,
                solid_color_tga((0.5, 0.5, 1.0, 1.0)),
                extension=".tga",
            ),
        ),
    )


def _model(
        *,
        material: AuthoringMaterial | None = None,
        document: ModelDocument | None = None,
) -> ExtractedAuthoringModel:
    value = material or _material()
    return ExtractedAuthoringModel(
        "crate",
        document or _document(),
        b"",
        (value,),
    )


def _source1_model(
        *,
        collision: Source1AuthoringCollision | None = None,
        lods: tuple[Source1AuthoringLod, ...] = (),
) -> Source1ExtractedAuthoringModel:
    model = _model()
    return Source1ExtractedAuthoringModel(
        model.name,
        model.document,
        model.fbx_payload,
        model.materials,
        collision,
        lods,
    )


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_serializers_match_source1_golden_contract():
    material = _material()
    color_path = ContentRelativePath(
        "materialsrc/models/sourceio/crate/crate_base_color.tga"
    )
    normal_path = ContentRelativePath(
        "materialsrc/models/sourceio/crate/crate_normal.tga"
    )

    assert serialize_smd(_document(), {"crate"}) == _fixture("sourceio.smd")
    assert serialize_qc(
        ContentRelativePath("models/sourceio/crate/crate.mdl"),
        "crate.smd",
        ContentRelativePath("materials/models/sourceio/crate"),
    ) == _fixture("sourceio.qc")
    assert serialize_vmt(
        material,
        color_path,
        normal_texture=normal_path,
    ) == _fixture("sourceio.vmt")
    assert serialize_vtex_settings(
        TextureRole.BASE_COLOR,
        alpha_mode=MaterialAlphaMode.MASK,
    ) == _fixture("sourceio_base_color.txt")
    assert serialize_vtex_settings(
        TextureRole.NORMAL,
    ) == _fixture("sourceio_normal.txt")


def test_source1_phong_and_existing_vmt_patch_are_explicit():
    material = Source1AuthoringMaterial(
        "crate",
        phong=Source1PhongParameters(
            exponent=16.0,
            boost=2.0,
            tint=(0.25, 0.5, 0.75),
            fresnel_ranges=(0.0, 0.5, 1.0),
        ),
    )

    vmt = serialize_vmt(
        material,
        ContentRelativePath(
            "materialsrc/models/sourceio/crate/crate_base_color.tga"
        ),
    )

    assert '"$phong" "1"' in vmt
    assert '"$phongexponent" "16"' in vmt
    assert '"$phongboost" "2"' in vmt
    assert '"$phongtint" "[0.25 0.5 0.75]"' in vmt
    assert '"$phongfresnelranges" "[0 0.5 1]"' in vmt
    assert serialize_vmt_patch(
        ContentRelativePath("models/props/imported.vmt")
    ) == (
        '"Patch"\n'
        "{\n"
        '    "include" "models/props/imported.vmt"\n'
        "}\n"
    )


def test_typed_vmt_parameters_and_explicit_scalar_pbr_conversion(tmp_path):
    phong = source1_pbr_to_phong(0.5, 0.5)
    assert phong.exponent == pytest.approx(30.0)
    assert phong.boost == pytest.approx(0.625)
    assert phong.albedo_tint
    material = Source1AuthoringMaterial(
        "crate",
        metallic=0.5,
        roughness=0.5,
        phong=Source1PhongParameters(
            exponent=phong.exponent,
            boost=phong.boost,
            albedo_tint=True,
            rim_light=True,
            rim_exponent=4.0,
            rim_boost=2.0,
        ),
        parameters=Source1MaterialParameters(
            self_illum=True,
            self_illum_tint=(0.25, 0.5, 0.75),
            envmap="env_cubemap",
            envmap_tint=(0.5, 0.5, 0.5),
            envmap_contrast=0.25,
            envmap_saturation=0.75,
        ),
        pbr_converted=True,
    )

    output = Source1AuthoringBackend().build(
        Source1AuthoringRequest(_target(tmp_path), "crate"),
        (_model(material=material),),
    )

    assert not output.report.has_errors
    vmt = next(
        artifact.payload for artifact in output.artifacts
        if artifact.kind == Source1ArtifactKind.MATERIAL
    )
    assert '"$phongexponent" "30"' in vmt
    assert '"$phongboost" "0.625"' in vmt
    assert '"$phongalbedotint" "1"' in vmt
    assert '"$rimlight" "1"' in vmt
    assert '"$rimlightexponent" "4"' in vmt
    assert '"$rimlightboost" "2"' in vmt
    assert '"$selfillum" "1"' in vmt
    assert '"$selfillumtint" "[0.25 0.5 0.75]"' in vmt
    assert '"$envmap" "env_cubemap"' in vmt
    assert '"$envmaptint" "[0.5 0.5 0.5]"' in vmt
    assert '"$envmapcontrast" "0.25"' in vmt
    assert '"$envmapsaturation" "0.75"' in vmt


def test_typed_vmt_parameters_reject_unsafe_dependencies():
    with pytest.raises(ValueError, match="requires self-illumination"):
        Source1MaterialParameters(self_illum_tint=(1, 1, 1))
    with pytest.raises(ValueError, match="safe material-relative"):
        Source1MaterialParameters(envmap="../escape.vtf")
    with pytest.raises(ValueError, match="require an envmap"):
        Source1MaterialParameters(envmap_contrast=1.0)
    with pytest.raises(ValueError, match="require rim lighting"):
        Source1PhongParameters(rim_boost=1.0)


def test_backend_reuses_existing_vmt_through_local_patch(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    material = Source1AuthoringMaterial(
        "crate",
        existing_vmt=ContentRelativePath("models/props/imported.vmt"),
    )

    output = Source1AuthoringBackend().build(
        request,
        (_model(material=material),),
    )

    assert not output.report.has_errors
    assert len(output.compile_steps) == 1
    patch = next(
        artifact for artifact in output.artifacts
        if artifact.kind == Source1ArtifactKind.MATERIAL
    )
    assert patch.payload == serialize_vmt_patch(material.existing_vmt)
    assert all(
        artifact.kind not in {
            Source1ArtifactKind.TEXTURE_SOURCE,
            Source1ArtifactKind.TEXTURE_SETTINGS,
        }
        for artifact in output.artifacts
    )


def test_backend_plans_dual_root_editable_bundle(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")

    output = Source1AuthoringBackend().build(request, (_model(),))

    assert not output.report.has_errors
    assert output.model_paths == (
        ContentRelativePath("models/sourceio/crate/crate.mdl"),
    )
    assert [
        (artifact.root.value, artifact.path.as_posix())
        for artifact in output.artifacts
    ] == [
        ("content", "materialsrc/models/sourceio/crate/crate_base_color.tga"),
        ("content", "materialsrc/models/sourceio/crate/crate_base_color.txt"),
        ("content", "materialsrc/models/sourceio/crate/crate_normal.tga"),
        ("content", "materialsrc/models/sourceio/crate/crate_normal.txt"),
        ("content", "modelsrc/sourceio/crate/crate.qc"),
        ("content", "modelsrc/sourceio/crate/crate.smd"),
        ("game", "materials/models/sourceio/crate/crate.vmt"),
    ]
    assert [
        (
            step.compiler.value,
            step.source.as_posix(),
            tuple(output.path.as_posix() for output in step.expected_outputs),
        )
        for step in output.compile_steps
    ] == [
        (
            "vtex",
            "materialsrc/models/sourceio/crate/crate_base_color.tga",
            ("materials/models/sourceio/crate/crate_base_color.vtf",),
        ),
        (
            "vtex",
            "materialsrc/models/sourceio/crate/crate_normal.tga",
            ("materials/models/sourceio/crate/crate_normal.vtf",),
        ),
        (
            "studiomdl",
            "modelsrc/sourceio/crate/crate.qc",
            (
                "models/sourceio/crate/crate.mdl",
                "models/sourceio/crate/crate.vvd",
                "models/sourceio/crate/crate.dx90.vtx",
            ),
        ),
    ]
    assert all(
        "crate_alpha" not in artifact.path.as_posix()
        for artifact in output.artifacts
    )


def test_backend_authors_collision_lods_and_phy_output(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    model = _source1_model(
        collision=Source1AuthoringCollision(
            _document(material="collision", uv=False),
            mass=12.5,
        ),
        lods=(
            Source1AuthoringLod(40, _document()),
            Source1AuthoringLod(15, _document()),
        ),
    )

    output = Source1AuthoringBackend().build(request, (model,))

    assert not output.report.has_errors
    assert tuple(lod.distance for lod in model.lods) == (15.0, 40.0)
    model_sources = {
        artifact.path.name: artifact.payload
        for artifact in output.artifacts
        if artifact.kind == Source1ArtifactKind.MODEL_SOURCE
    }
    assert set(model_sources) == {
        "crate.smd",
        "crate_collision.smd",
        "crate_lod1.smd",
        "crate_lod2.smd",
    }
    assert "  0 0 0 0 0 0 1 0 1" in model_sources["crate_collision.smd"]
    qc = next(
        artifact.payload
        for artifact in output.artifacts
        if artifact.kind == Source1ArtifactKind.MODEL_SCRIPT
    )
    assert (
        '$lod 15\n'
        '{\n'
        '    replacemodel "crate.smd" "crate_lod1.smd"\n'
        '}\n'
        '$lod 40\n'
        '{\n'
        '    replacemodel "crate.smd" "crate_lod2.smd"\n'
        '}\n'
        '$collisionmodel "crate_collision.smd"\n'
        '{\n'
        '    $mass 12.5\n'
        '    $concave\n'
        '}'
    ) in qc
    studiomdl = next(
        step for step in output.compile_steps
        if step.compiler.value == "studiomdl"
    )
    assert tuple(
        item.path.name for item in studiomdl.expected_outputs
    ) == (
        "crate.mdl",
        "crate.vvd",
        "crate.dx90.vtx",
        "crate.phy",
    )


def test_lod_material_mismatch_is_blocked(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    model = _source1_model(
        lods=(
            Source1AuthoringLod(
                20,
                _document(material="other"),
            ),
        ),
    )

    output = Source1AuthoringBackend().build(request, (model,))

    assert {
        diagnostic.code for diagnostic in output.report
    } >= {
        "authoring.source1.lod.materials_mismatch",
        "authoring.source1.lod.smd.invalid",
    }


def test_collision_and_lod_specs_reject_invalid_values():
    document = _document()

    with pytest.raises(ValueError, match="finite and positive"):
        Source1AuthoringCollision(document, mass=0)
    with pytest.raises(ValueError, match="finite and positive"):
        Source1AuthoringLod(0, document)
    with pytest.raises(ValueError, match="unique"):
        Source1ExtractedAuthoringModel(
            "crate",
            document,
            b"",
            (_material(),),
            lods=(
                Source1AuthoringLod(10, document),
                Source1AuthoringLod(10, document),
            ),
        )


def test_backend_authors_typed_bodygroups_skins_attachments_and_hitboxes(
        tmp_path,
):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    model = _model()
    source1_model = Source1ExtractedAuthoringModel(
        model.name,
        model.document,
        model.fbx_payload,
        model.materials,
        bodygroups=(
            Source1Bodygroup(
                "accessory",
                (
                    Source1BodygroupChoice("blank", None),
                    Source1BodygroupChoice("crate", _document()),
                ),
            ),
        ),
        skin_families=(
            Source1SkinFamily(("crate",)),
            Source1SkinFamily(("crate_blue",)),
        ),
        attachments=(
            Source1Attachment(
                "muzzle",
                "root",
                (1.0, 2.0, 3.0),
                (10.0, 20.0, 30.0),
            ),
        ),
        hitboxes=(
            Source1Hitbox(
                1,
                "root",
                (-1.0, -2.0, -3.0),
                (1.0, 2.0, 3.0),
            ),
        ),
    )

    output = Source1AuthoringBackend().build(request, (source1_model,))

    assert not output.report.has_errors
    paths = {artifact.path.as_posix() for artifact in output.artifacts}
    assert (
        "modelsrc/sourceio/crate/crate_bodygroup1_crate.smd"
        in paths
    )
    qc = next(
        artifact.payload for artifact in output.artifacts
        if artifact.kind == Source1ArtifactKind.MODEL_SCRIPT
    )
    assert '$bodygroup "accessory"' in qc
    assert "    blank" in qc
    assert '    studio "crate_bodygroup1_crate.smd"' in qc
    assert '$texturegroup "skinfamilies"' in qc
    assert '    { "crate" }' in qc
    assert '    { "crate_blue" }' in qc
    assert (
        '$attachment "muzzle" "root" 1 2 3 rotate 10 20 30'
        in qc
    )
    assert '$hbox 1 "root" -1 -2 -3 1 2 3' in qc


def test_typed_qc_models_reject_unsafe_or_inconsistent_values():
    with pytest.raises(ValueError, match="lowercase ASCII"):
        Source1BodygroupChoice("../escape", None)
    with pytest.raises(ValueError, match="bare VMT"):
        Source1SkinFamily(("materials/crate.vmt",))
    with pytest.raises(ValueError, match="minimum"):
        Source1Hitbox(0, "root", (1, 0, 0), (0, 1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        Source1Hitbox(-1, "root", (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="finite"):
        Source1Attachment("muzzle", "root", (0, 0, float("nan")))


def test_backend_rejects_mismatched_skin_family_width(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    model = _model()
    source1_model = Source1ExtractedAuthoringModel(
        model.name,
        model.document,
        model.fbx_payload,
        model.materials,
        skin_families=(
            Source1SkinFamily(("crate", "extra")),
        ),
    )

    output = Source1AuthoringBackend().build(request, (source1_model,))

    assert "authoring.source1.skin.width_mismatch" in {
        diagnostic.code for diagnostic in output.report
    }


def test_backend_authors_weighted_skeletal_smd_without_staticprop(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(
        target,
        "crate",
        export_skeleton=True,
    )
    document = _document()
    for vertex in document.meshes[0].vertices:
        vertex.corner.bone_indices = (0, 1)
        vertex.corner.bone_weights = (0.25, 0.75)
    skeleton = Source1Skeleton((
        Source1Bone("root", -1, (0, 0, 0), (0, 0, 0)),
        Source1Bone("child", 0, (0, 0, 8), (0, 0, 0)),
    ))
    model = Source1ExtractedAuthoringModel(
        "crate",
        document,
        b"",
        (_material(),),
        skeleton=skeleton,
    )

    output = Source1AuthoringBackend().build(request, (model,))

    assert not output.report.has_errors
    source = next(
        artifact.payload for artifact in output.artifacts
        if artifact.path.suffix == ".smd"
    )
    parsed = parse_smd(source)
    assert [node.name for node in parsed.nodes] == ["root", "child"]
    assert "  1 0 0 0 0 0 1 0 0 2 1 0.75 0 0.25" in source
    qc = next(
        artifact.payload for artifact in output.artifacts
        if artifact.kind == Source1ArtifactKind.MODEL_SCRIPT
    )
    assert "$staticprop" not in qc


def test_service_writes_manifest_report_and_artifacts(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    backend = Source1AuthoringBackend().build(request, (_model(),))
    service = Source1AuthoringService()

    plan = service.plan(request, backend)
    result = service.execute(plan)

    assert len(result.written_paths) == 9
    assert target.content_path(
        "modelsrc/sourceio/crate/crate.smd"
    ).read_text(encoding="utf-8") == _fixture("sourceio.smd")
    assert target.game_path(
        "materials/models/sourceio/crate/crate.vmt"
    ).read_text(encoding="utf-8") == _fixture("sourceio.vmt")
    manifest_text = target.content_path(result.manifest_path).read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in manifest_text
    manifest = Source1AuthoringManifest.from_json(manifest_text)
    manifest.validate_target(target)
    assert manifest.to_json() == plan.manifest.to_json()
    data = json.loads(manifest_text)
    assert data["schema_version"] == 2
    assert data["target"] == {
        "content_directory": "tf_content",
        "game_directory": "tf",
    }
    assert len(data["artifacts"]) == 8
    assert len(data["compile"]) == 3


def test_preflight_errors_block_every_write(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "Bad Name")
    backend = Source1AuthoringBackend().build(request, (_model(),))
    plan = Source1AuthoringService().plan(request, backend)

    with pytest.raises(ExportBlockedError):
        plan.write()

    assert not target.content_path(
        "modelsrc/sourceio/Bad Name/crate.smd"
    ).exists()
    assert not target.content_path(
        "sourceio/manifests/Bad Name.sourceio.json"
    ).exists()


def test_existing_artifacts_are_protected_without_overwrite(tmp_path):
    target = _target(tmp_path)
    existing = target.game_path("materials/models/sourceio/crate/crate.vmt")
    existing.parent.mkdir(parents=True)
    existing.write_text("existing", encoding="utf-8")
    request = Source1AuthoringRequest(target, "crate")
    backend = Source1AuthoringBackend().build(request, (_model(),))

    plan = Source1AuthoringService().plan(request, backend)

    assert "authoring.output.exists" in {
        diagnostic.code for diagnostic in plan.report
    }
    with pytest.raises(ExportBlockedError):
        plan.write()
    assert existing.read_text(encoding="utf-8") == "existing"


def test_existing_compiled_outputs_are_protected_when_compiling(tmp_path):
    target = _target(tmp_path)
    existing = target.game_path(
        "materials/models/sourceio/crate/crate_base_color.vtf"
    )
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")
    request = Source1AuthoringRequest(
        target,
        "crate",
        compile_after_export=True,
    )
    backend = Source1AuthoringBackend().build(request, (_model(),))

    plan = Source1AuthoringService().plan(request, backend)

    assert "authoring.output.exists" in {
        diagnostic.code for diagnostic in plan.report
    }
    with pytest.raises(ExportBlockedError):
        plan.write()
    assert existing.read_bytes() == b"existing"


def test_pbr_conversion_and_missing_uv_are_blocked(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    model = _model(
        material=_material(metallic=0.25, roughness=0.75),
        document=_document(uv=False),
    )

    output = Source1AuthoringBackend().build(request, (model,))

    assert {
        diagnostic.code for diagnostic in output.report
    } >= {
        "authoring.source1.material.pbr_conversion_unsupported",
        "authoring.source1.smd.invalid",
    }


def test_opaque_material_cannot_carry_effective_alpha(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(target, "crate")
    material = AuthoringMaterial(
        "crate",
        base_color=(1.0, 1.0, 1.0, 0.5),
        alpha_mode=MaterialAlphaMode.OPAQUE,
        textures=(
            AuthoringTexture(
                "crate_base_color",
                TextureRole.BASE_COLOR,
                solid_color_tga((1.0, 1.0, 1.0, 1.0)),
                extension=".tga",
            ),
        ),
    )

    output = Source1AuthoringBackend().build(
        request,
        (_model(material=material),),
    )

    assert "authoring.source1.material.alpha_mode_mismatch" in {
        diagnostic.code for diagnostic in output.report
    }


def test_blended_material_uses_translucent_not_alpha_test():
    material = _material(alpha_mode=MaterialAlphaMode.BLEND)

    value = serialize_vmt(
        material,
        ContentRelativePath(
            "materialsrc/models/sourceio/crate/crate_base_color.tga"
        ),
    )

    assert '"$translucent" "1"' in value
    assert "$alphatest" not in value


def test_material_source_and_runtime_directories_must_match(tmp_path):
    target = _target(tmp_path)
    request = Source1AuthoringRequest(
        target,
        "crate",
        material_source_directory=ContentRelativePath("materialsrc/sourceio"),
        material_directory=ContentRelativePath("materials/models/sourceio"),
    )
    plan = Source1AuthoringService().plan(
        request,
        Source1AuthoringBackend().build(request, (_model(),)),
    )

    assert "authoring.source1.material_directories.mismatch" in {
        diagnostic.code for diagnostic in plan.report
    }


def test_target_requires_existing_roots_and_gameinfo(tmp_path):
    game_root = tmp_path / "tf"
    content_root = tmp_path / "tf_content"
    game_root.mkdir()
    content_root.mkdir()

    with pytest.raises(Source1AuthoringTargetError, match="gameinfo.txt"):
        Source1AuthoringTarget(game_root, content_root)
    with pytest.raises(Source1AuthoringTargetError, match="unavailable"):
        Source1AuthoringTarget(tmp_path / "missing", content_root)


def test_manifest_target_identity_is_validated(tmp_path):
    first = _target(tmp_path, "tf")
    request = Source1AuthoringRequest(first, "crate")
    backend = Source1AuthoringBackend().build(request, (_model(),))
    manifest = Source1AuthoringManifest.build(
        request,
        backend.artifacts,
        backend.compile_steps,
        backend.model_paths,
    )
    second = _target(tmp_path, "hl2")

    with pytest.raises(Source1AuthoringManifestError, match="does not match"):
        manifest.validate_target(second)


def test_artifact_root_is_typed():
    artifact = Source1AuthoringArtifact(
        "content",
        "sourceio/reports/crate.json",
        "report",
        "",
    )

    assert artifact.root is Source1ArtifactRoot.CONTENT
    assert artifact.kind is Source1ArtifactKind.REPORT


def test_staging_restores_overwritten_files_after_partial_failure(
        tmp_path,
        monkeypatch,
):
    target = _target(tmp_path)
    existing = target.game_path("materials/sourceio/existing.vmt")
    existing.parent.mkdir(parents=True)
    existing.write_text("original", encoding="utf-8")
    artifacts = (
        Source1AuthoringArtifact(
            Source1ArtifactRoot.GAME,
            "materials/sourceio/existing.vmt",
            Source1ArtifactKind.MATERIAL,
            "replacement",
        ),
        Source1AuthoringArtifact(
            Source1ArtifactRoot.CONTENT,
            "modelsrc/sourceio/new.smd",
            Source1ArtifactKind.MODEL_SOURCE,
            "new",
        ),
    )
    real_replace = __import__("os").replace

    def fail_new_source(source, destination):
        if Path(destination).name == "new.smd":
            raise OSError("synthetic commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "SourceIO.library.shared.authoring.staging.os.replace",
        fail_new_source,
    )

    with pytest.raises(OSError, match="synthetic"):
        stage_source1_artifacts(target, artifacts, overwrite=True)

    assert existing.read_text(encoding="utf-8") == "original"
    assert not target.content_path("modelsrc/sourceio/new.smd").exists()
    assert not list(tmp_path.rglob("*.sourceio-stage"))
    assert not list(tmp_path.rglob("*.sourceio-backup"))
