import struct
from pathlib import Path

import numpy as np

from SourceIO.library.models.phy.phy import Phy
from SourceIO.library.utils import FileBuffer, MemoryBuffer


def _legacy_phy(marker: bytes = b"\0\0\0\0") -> bytes:
    data = bytearray(177)
    struct.pack_into("<4I", data, 0, 16, 0, 1, 123)
    struct.pack_into("<I", data, 16, 156)
    struct.pack_into("<7f4I4s", data, 20, *([0.0] * 7), 0, 48, 0, 0, marker)
    struct.pack_into("<2i4f4B", data, 68, 0, 28, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0)
    struct.pack_into("<3i2h", data, 96, 32, 0, 0, 1, 0)
    struct.pack_into("<i6h", data, 112, 0, 0, 0, 1, 0, 2, 0)
    vertices = np.array(
        [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
        ],
        dtype=np.float32,
    )
    data[128:176] = vertices.tobytes()
    return bytes(data)


def test_wrapperless_phy_accepts_zero_beta_marker():
    buffer = MemoryBuffer(_legacy_phy())

    phy = Phy.from_buffer(buffer)

    assert phy.header.checksum == 123
    assert len(phy.solids) == 1
    solid = phy.solids[0]
    assert solid.version == 0
    assert solid.collision_model.root_tree.convex_leaf is not None
    vertices = solid.collision_model.get_vertex_data(
        buffer,
        solid.collision_model.root_tree.convex_leaf,
        3,
    )
    assert vertices.tolist() == [
        [1.0, 3.0, 2.0],
        [4.0, 6.0, 5.0],
        [7.0, 9.0, 8.0],
    ]


def test_modern_vphy_sample_still_parses():
    sample = Path(__file__).parents[2] / "samples" / "dog.phy"

    with FileBuffer(sample) as buffer:
        phy = Phy.from_buffer(buffer)

    assert len(phy.solids) == 16
    assert all(solid.version == 0x100 for solid in phy.solids)
