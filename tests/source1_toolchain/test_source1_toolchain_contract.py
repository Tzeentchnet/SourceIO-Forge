from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
from SourceIO.library.source1.authoring import (
    Source1ArtifactRoot,
    Source1Attachment,
    Source1AuthoringBackend,
    Source1AuthoringMaterial,
    Source1AuthoringRequest,
    Source1AuthoringService,
    Source1AuthoringTarget,
    Source1Bodygroup,
    Source1BodygroupChoice,
    Source1Bone,
    Source1ExtractedAuthoringModel,
    Source1Hitbox,
    Source1MaterialParameters,
    Source1Skeleton,
    Source1SkinFamily,
    source1_pbr_to_phong,
)
from SourceIO.library.source1.vtf import load_texture
from SourceIO.library.source2.authoring import (
    AuthoringTexture,
    MaterialAlphaMode,
    TextureRole,
    solid_color_tga,
)
from SourceIO.library.source2.export import (
    CornerData,
    MeshFace,
    MeshVertex,
    ModelDocument,
    StaticMesh,
)
from SourceIO.library.utils import FileBuffer


def _configured_target() -> tuple[Source1AuthoringTarget, Path]:
    install_value = os.environ.get("SOURCEIO_SOURCE1_TOOLCHAIN_ROOT")
    if not install_value:
        pytest.skip("SOURCEIO_SOURCE1_TOOLCHAIN_ROOT is not configured")
    install_root = Path(install_value).expanduser().resolve()
    game_root = Path(
        os.environ.get("SOURCEIO_SOURCE1_GAME_ROOT", install_root / "tf")
    ).expanduser().resolve()
    content_root = Path(
        os.environ.get("SOURCEIO_SOURCE1_CONTENT_ROOT", install_root)
    ).expanduser().resolve()
    return Source1AuthoringTarget(game_root, content_root), install_root / "bin"


def _model(name: str) -> Source1ExtractedAuthoringModel:
    vertices = (
        MeshVertex(
            (0.0, 0.0, 0.0),
            CornerData(
                normal=(0.0, 0.0, 1.0),
                texcoords=((0.0, 1.0),),
                bone_indices=(0,),
                bone_weights=(1.0,),
            ),
        ),
        MeshVertex(
            (16.0, 0.0, 0.0),
            CornerData(
                normal=(0.0, 0.0, 1.0),
                texcoords=((1.0, 1.0),),
                bone_indices=(1,),
                bone_weights=(1.0,),
            ),
        ),
        MeshVertex(
            (0.0, 16.0, 0.0),
            CornerData(
                normal=(0.0, 0.0, 1.0),
                texcoords=((0.0, 0.0),),
                bone_indices=(1,),
                bone_weights=(1.0,),
            ),
        ),
    )
    document = ModelDocument(
        name,
        (StaticMesh(
            f"{name}_mesh",
            vertices,
            (MeshFace((0, 1, 2), "probe"),),
        ),),
    )
    material = Source1AuthoringMaterial(
        "probe",
        metallic=0.5,
        roughness=0.5,
        alpha_mode=MaterialAlphaMode.MASK,
        alpha_cutoff=0.4,
        textures=(
            AuthoringTexture(
                "probe_base_color",
                TextureRole.BASE_COLOR,
                solid_color_tga((0.8, 0.2, 0.1, 0.5)),
                extension=".tga",
                srgb=True,
            ),
            AuthoringTexture(
                "probe_alpha",
                TextureRole.ALPHA,
                solid_color_tga((0.5, 0.5, 0.5, 1.0)),
                extension=".tga",
            ),
            AuthoringTexture(
                "probe_normal",
                TextureRole.NORMAL,
                solid_color_tga((0.5, 0.5, 1.0, 1.0)),
                extension=".tga",
            ),
        ),
        phong=source1_pbr_to_phong(0.5, 0.5),
        parameters=Source1MaterialParameters(
            self_illum=True,
            self_illum_tint=(0.8, 0.2, 0.1),
            envmap="env_cubemap",
        ),
        pbr_converted=True,
    )
    skeleton = Source1Skeleton((
        Source1Bone("root", -1, (0, 0, 0), (0, 0, 0)),
        Source1Bone("child", 0, (0, 0, 8), (0, 0, 0)),
    ))
    return Source1ExtractedAuthoringModel(
        name,
        document,
        b"",
        (material,),
        bodygroups=(
            Source1Bodygroup(
                "accessory",
                (
                    Source1BodygroupChoice("blank", None),
                    Source1BodygroupChoice("probe", document),
                ),
            ),
        ),
        skin_families=(
            Source1SkinFamily(("probe",)),
            Source1SkinFamily(("probe_blue",)),
        ),
        attachments=(
            Source1Attachment("muzzle", "child", (0, 0, 1)),
        ),
        hitboxes=(
            Source1Hitbox(1, "child", (-1, -1, -1), (1, 1, 1)),
        ),
        skeleton=skeleton,
    )


def _remove_tree(path: Path):
    if path.is_dir():
        shutil.rmtree(path)


def test_source1_bundle_compiles_reimports_and_recompiles():
    target, tool_root = _configured_target()
    asset_name = f"s1test_{uuid.uuid4().hex[:12]}"
    request = Source1AuthoringRequest(
        target,
        asset_name,
        compile_after_export=True,
        export_skeleton=True,
    )
    service = Source1AuthoringService()
    backend = Source1AuthoringBackend().build(
        request,
        (_model(asset_name),),
    )
    plan = service.plan(request, backend)
    model_directory = target.game_path(f"models/sourceio/{asset_name}")
    material_directory = target.game_path(
        f"materials/models/sourceio/{asset_name}"
    )
    model_source_directory = target.content_path(
        f"modelsrc/sourceio/{asset_name}"
    )
    material_source_directory = target.content_path(
        f"materialsrc/models/sourceio/{asset_name}"
    )
    try:
        result = service.execute(plan, tool_root=tool_root)

        assert len(result.compiler_results) == 3
        mdl_path = model_directory / f"{asset_name}.mdl"
        with FileBuffer(mdl_path) as buffer:
            mdl = MdlV49.from_buffer(buffer)
        assert mdl.header.name.replace("\\", "/") == (
            f"sourceio/{asset_name}/{asset_name}.mdl"
        )
        assert {bone.name for bone in mdl.bones} >= {"root", "child"}
        for texture_name in ("probe_base_color", "probe_normal"):
            with (material_directory / f"{texture_name}.vtf").open("rb") as stream:
                pixels, height, width = load_texture(stream, normalized=False)
            assert (width, height) == (4, 4)
            assert pixels.shape == (4, 4, 4)

        recompiled = service.recompile(
            target,
            result.manifest_path,
            tool_root=tool_root,
        )
        assert len(recompiled) == 3
    finally:
        _remove_tree(model_directory)
        _remove_tree(material_directory)
        _remove_tree(model_source_directory)
        _remove_tree(material_source_directory)
        for artifact in plan.artifacts:
            if (
                    artifact.root == Source1ArtifactRoot.CONTENT
                    and artifact.kind.value in {"manifest", "report", "log"}
            ):
                path = target.content_path(artifact.path)
                if path.is_file():
                    path.unlink()
        log_path = target.content_path(
            f"sourceio/logs/{asset_name}.compile.log"
        )
        if log_path.is_file():
            log_path.unlink()
