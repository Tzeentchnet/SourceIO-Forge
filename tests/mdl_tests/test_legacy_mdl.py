import struct
from types import SimpleNamespace

import numpy as np
import pytest

from SourceIO.library.models.mdl.structs.bone import Bone
from SourceIO.library.models.mdl.structs.flex import Flex, VertexAminationType
from SourceIO.library.models.mdl.structs.header import MdlHeaderV36
from SourceIO.library.models.mdl.structs.legacy_animation import LegacyAnimDesc
from SourceIO.library.models.mdl.structs.legacy_sequence import LegacySequence
from SourceIO.library.models.mdl.structs.mesh import MeshV36Plus
from SourceIO.library.models.mdl.structs.model import ModelV36Plus, VERTEX_DTYPE
from SourceIO.library.models.mdl.v36.mdl_file import MdlV36
from SourceIO.library.utils.math_utilities import euler_to_quat
from SourceIO.library.models.vtx.v6.structs.header import Header as VtxHeader
from SourceIO.library.utils import MemoryBuffer


def test_v39_header_accounts_for_beta_sequence_fields():
    data = bytearray(512)
    struct.pack_into("<4sii64sI", data, 0, b"IDST", 39, -1, b"beta.mdl", len(data))
    struct.pack_into("<I", data, 152, 17)
    struct.pack_into("<4I", data, 196, 1, 240, 1, 280)
    struct.pack_into("<3I", data, 212, 0, 1, 320)
    struct.pack_into("<2I", data, 224, 1, 400)
    struct.pack_into("<2I", data, 232, 1, 432)
    struct.pack_into("<3I", data, 240, 1, 1, 436)
    struct.pack_into("<2I", data, 252, 1, 440)
    struct.pack_into("<I", data, 324, 480)
    struct.pack_into("<fI", data, 344, 4000.0, 1)
    data[480:486] = b"metal\0"

    header = MdlHeaderV36.from_buffer(MemoryBuffer(data))

    assert header.version == 39
    assert (header.v39_sequence_aux_count, header.v39_sequence_aux_offset) == (1, 240)
    assert (header.v39_sequence_count, header.v39_sequence_offset) == (1, 280)
    assert (header.sequence_group_count, header.sequence_group_offset) == (1, 320)
    assert (header.texture_count, header.texture_offset) == (1, 400)
    assert (header.texture_path_count, header.texture_path_offset) == (1, 432)
    assert (
        header.skin_reference_count,
        header.skin_family_count,
        header.skin_family_offset,
    ) == (1, 1, 436)
    assert (header.body_part_count, header.body_part_offset) == (1, 440)
    assert header.surface_prop == "metal"
    assert header.mass == 4000.0
    assert header.contents == 1


def _animation_bone(name, position, rotation, position_scale, rotation_scale, quat=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        name=name,
        position=position,
        rotation=rotation,
        position_scale=position_scale,
        rotation_scale=rotation_scale,
        quat=quat,
    )


def test_v35_fixed_animation_records_decode_raw_and_rle_channels():
    data = bytearray(512)
    struct.pack_into(
        "<If4I6f3I8I",
        data,
        0,
        400,
        30.0,
        1,
        3,
        0,
        0,
        *([0.0] * 6),
        92,
        0,
        0,
        *([0] * 8),
    )
    struct.pack_into("<I3i3iI", data, 92, 3, 64, 0, 0, 70, 0, 0, 0)
    struct.pack_into("<I3f4f", data, 124, 0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0)
    struct.pack_into("<BB2h", data, 156, 2, 3, 10, 20)
    struct.pack_into("<BB3h", data, 162, 3, 3, 100, 200, 300)
    data[400:406] = b"@test\0"

    buffer = MemoryBuffer(data)
    desc = LegacyAnimDesc.from_buffer(buffer, 35)
    bones = [
        _animation_bone("animated", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3),
                        (0.5, 1.0, 1.0), (0.001, 1.0, 1.0)),
        _animation_bone("raw", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    ]

    frames = desc.read_animations(buffer, bones)

    assert desc.name == "@test"
    assert desc.frame_count == 3
    np.testing.assert_allclose(frames["animated"]["pos"], ((6.0, 2.0, 3.0),
                                                           (11.0, 2.0, 3.0),
                                                           (11.0, 2.0, 3.0)))
    expected_eulers = np.array(((0.2, 0.2, 0.3), (0.3, 0.2, 0.3), (0.4, 0.2, 0.3)), np.float32)
    np.testing.assert_allclose(frames["animated"]["rot"], euler_to_quat(expected_eulers), atol=1e-6)
    np.testing.assert_allclose(frames["raw"]["pos"], ((4.0, 5.0, 6.0),) * 3)
    np.testing.assert_allclose(frames["raw"]["rot"], ((0.0, 0.0, 0.0, 1.0),) * 3)


def test_v39_animation_descriptor_uses_sparse_records():
    data = bytearray(320)
    struct.pack_into(
        "<If4I6f3I8I",
        data,
        0,
        240,
        30.0,
        0,
        2,
        0,
        0,
        *([0.0] * 6),
        92,
        0,
        0,
        *([0] * 8),
    )
    struct.pack_into("<BBh3hBBh", data, 92, 0, 4, 0, 6, 0, 0, 1, 2, 4)
    data[240:246] = b"@idle\0"
    bone = _animation_bone("root", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3),
                           (0.5, 1.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    buffer = MemoryBuffer(data)

    desc = LegacyAnimDesc.from_buffer(buffer, 39)
    frames = desc.read_animations(buffer, [bone])

    np.testing.assert_allclose(frames["root"]["pos"], ((3.0, 2.0, 3.0), (3.0, 2.0, 3.0)))
    np.testing.assert_allclose(frames["root"]["rot"], ((0.0, 0.0, 0.0, 1.0),) * 2)


def test_v39_sparse_animation_accepts_empty_terminator():
    data = bytearray(256)
    struct.pack_into(
        "<If4I6f3I8I",
        data,
        0,
        200,
        30.0,
        0,
        1,
        0,
        0,
        *([0.0] * 6),
        92,
        0,
        0,
        *([0] * 8),
    )
    struct.pack_into("<BBh", data, 92, 255, 0, 0)
    data[200:206] = b"@idle\0"
    buffer = MemoryBuffer(data)

    desc = LegacyAnimDesc.from_buffer(buffer, 39)

    assert desc.read_animations(buffer, [_animation_bone(
        "root", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    )]) == {}


@pytest.mark.parametrize(("version", "parsed_size"), [(32, 188), (35, 700)])
def test_v32_v35_sequence_records_use_fixed_4284_byte_stride(version, parsed_size):
    data = bytearray(4600)
    struct.pack_into("<7i6f", data, 0, 4500, 0, 1, -1, 0, 0, 4284, *([0.0] * 6))
    if version == 32:
        struct.pack_into("<4i", data, 52, 3, 0, 0, 0)
    else:
        struct.pack_into("<I", data, 52, 1)
        struct.pack_into("<h", data, 56, 0)
    struct.pack_into("<f", data, 4284, 1.0)
    data[4500:4505] = b"idle\0"
    buffer = MemoryBuffer(data)

    sequence = LegacySequence.from_buffer(buffer, version)

    assert sequence.name == "idle"
    assert sequence.anim_desc_indices == [0]
    assert sequence.frame_count == (3 if version == 32 else None)
    assert sequence.get_bone_weights(buffer, 1) == [1.0]
    assert buffer.tell() == 4284
    assert parsed_size < 4284


def test_v39_sequence_uses_208_byte_pre_baseptr_layout():
    data = bytearray(400)
    struct.pack_into("<7i6f", data, 0, 300, 0, 0, -1, 0, 0, 208, *([0.0] * 6))
    struct.pack_into("<3I2I2i", data, 52, 1, 212, 0, 1, 1, -1, -1)
    struct.pack_into("<5I", data, 140, 0, 0, 0, 208, 0)
    struct.pack_into("<f", data, 208, 1.0)
    struct.pack_into("<h", data, 212, 0)
    data[300:305] = b"idle\0"
    buffer = MemoryBuffer(data)

    sequence = LegacySequence.from_buffer(buffer, 39)

    assert sequence.name == "idle"
    assert sequence.activity == -1
    assert sequence.blend_count == 1
    assert sequence.group_size == (1, 1)
    assert sequence.anim_desc_indices == [0]
    assert sequence.get_bone_weights(buffer, 1) == [1.0]
    assert buffer.tell() == 208


def test_v39_model_uses_transitional_sequence_table():
    data = bytearray(1800)
    struct.pack_into("<4sii64sI", data, 0, b"IDST", 39, -1, b"beta.mdl", len(data))
    struct.pack_into("<2I", data, 180, 1, 400)
    struct.pack_into("<2I", data, 188, 1, 392)
    struct.pack_into("<4I", data, 196, 1, 520, 1, 600)
    struct.pack_into(
        "<If4I6f3I8I",
        data,
        400,
        1000,
        30.0,
        0,
        1,
        0,
        0,
        *([0.0] * 6),
        92,
        0,
        0,
        *([0] * 8),
    )
    struct.pack_into("<BBh", data, 492, 255, 0, 0)
    struct.pack_into("<7i6f", data, 600, 900, 0, 0, -1, 0, 0, 208, *([0.0] * 6))
    struct.pack_into("<3I2I2i", data, 652, 1, 212, 0, 1, 1, -1, -1)
    struct.pack_into("<5I", data, 740, 0, 0, 0, 208, 0)
    struct.pack_into("<h", data, 812, 0)
    data[1400:1406] = b"@idle\0"
    data[1500:1505] = b"idle\0"

    mdl = MdlV36.from_buffer(MemoryBuffer(data))

    assert [desc.name for desc in mdl.anim_descs] == ["@idle"]
    assert [sequence.name for sequence in mdl.sequences] == ["idle"]
    assert mdl.sequences[0].anim_desc_indices == [0]


@pytest.mark.parametrize(("version", "record_size"), [(32, 196), (35, 196), (36, 196), (39, 216)])
def test_legacy_bones_keep_record_alignment(version, record_size):
    data = bytearray(record_size * 2 + 32)
    first_name = record_size * 2
    second_name = first_name + 6
    struct.pack_into("<i", data, 0, first_name)
    struct.pack_into("<i", data, record_size, second_name - record_size)
    if version <= 36:
        struct.pack_into("<4f", data, 164, 0.0, 0.0, 0.0, 1.0)
        struct.pack_into("<4f", data, record_size + 164, 0.0, 0.0, 0.0, 1.0)
    else:
        struct.pack_into("<4f", data, 44, 0.0, 0.0, 0.0, 1.0)
        struct.pack_into("<4f", data, record_size + 44, 0.0, 0.0, 0.0, 1.0)
    data[first_name:first_name + 6] = b"first\0"
    data[second_name:second_name + 7] = b"second\0"
    buffer = MemoryBuffer(data)

    first = Bone.from_buffer(buffer, version)
    second = Bone.from_buffer(buffer, version)

    assert first.name == "first"
    assert second.name == "second"
    assert buffer.tell() == record_size * 2


def test_v39_model_reads_embedded_vertices():
    vertices = np.zeros(1, dtype=VERTEX_DTYPE)
    vertices["vertex"][0] = (1.0, 2.0, 3.0)
    data = bytearray(140 + VERTEX_DTYPE.itemsize)
    data[:9] = b"beta.smd\0"
    struct.pack_into("<If9I", data, 64, 0, 4.0, 0, 140, 1, 140, 204, 0, 0, 0, 0)
    data[140:] = vertices.tobytes()
    buffer = MemoryBuffer(data)

    model = ModelV36Plus.from_buffer(buffer, 39)

    assert model.name == "beta.smd"
    assert model.vertex_count == 1
    assert model.vertices is not None
    assert model.vertices["vertex"][0].tolist() == [1.0, 2.0, 3.0]
    assert buffer.tell() == 140


def test_v39_mesh_and_flex_use_embedded_layout():
    mesh_buffer = MemoryBuffer(bytearray(68))
    mesh = MeshV36Plus.from_buffer(mesh_buffer, 39)
    flex_buffer = MemoryBuffer(bytearray(28))
    flex = Flex.from_buffer(flex_buffer, 39)

    assert mesh.flexes == []
    assert mesh_buffer.tell() == 68
    assert flex.partner_index is None
    assert flex.vertex_anim_type == VertexAminationType.NORMAL
    assert flex_buffer.tell() == 28


@pytest.mark.parametrize("max_bones_per_vertex", [2, 3])
def test_vtx_v6_accepts_beta_bone_limits(max_bones_per_vertex):
    data = struct.pack(
        "<IIHH6I",
        6,
        16,
        2,
        2,
        max_bones_per_vertex,
        123,
        1,
        0,
        0,
        0,
    )

    header = VtxHeader.from_buffer(MemoryBuffer(data))

    assert header.max_bones_per_vertex == max_bones_per_vertex


def test_vtx_v6_rejects_unknown_bone_limit():
    data = struct.pack("<IIHH6I", 6, 16, 2, 2, 4, 123, 1, 0, 0, 0)

    with pytest.raises(ValueError, match="Unsupported count of bones per vertex: 4"):
        VtxHeader.from_buffer(MemoryBuffer(data))
