import struct

import numpy as np
import pytest

from SourceIO.library.models.mdl.structs.bone import Bone
from SourceIO.library.models.mdl.structs.flex import Flex, VertexAminationType
from SourceIO.library.models.mdl.structs.header import MdlHeaderV36
from SourceIO.library.models.mdl.structs.mesh import MeshV36Plus
from SourceIO.library.models.mdl.structs.model import ModelV36Plus, VERTEX_DTYPE
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
