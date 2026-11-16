from dataclasses import dataclass
from enum import IntFlag

import numpy as np
import numpy.typing as npt

from ....shared.vector_types import Vector3
from ....utils import Buffer
from ....utils.math_utilities import euler_to_quat
from .bone import Bone
from .local_animation import ANIM_DTYPE, AnimDescFlags, StudioAnimationDecoder


class LegacyAnimBoneFlags(IntFlag):
    POSITION = 0x01
    ROTATION = 0x02


@dataclass(slots=True)
class LegacyAnimDesc(StudioAnimationDecoder):
    _entry_offset: int
    version: int
    name: str
    fps: float
    flags: AnimDescFlags
    frame_count: int
    movement_count: int
    movement_offset: int
    bbox: tuple[Vector3[float], Vector3[float]]
    anim_offset: int
    ikrule_count: int
    ikrule_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        if version not in (32, 35, 36, 39):
            raise ValueError(f"Unsupported legacy animation version {version}")

        entry_offset = buffer.tell()
        name_offset, fps, flags, frame_count, movement_count, movement_offset = buffer.read_fmt("If4I")
        bbox = buffer.read_fmt("3f"), buffer.read_fmt("3f")
        anim_offset, ikrule_count, ikrule_offset = buffer.read_fmt("3I")
        buffer.skip(8 * 4)

        name = ""
        if name_offset:
            name_position = entry_offset + name_offset
            if not 0 <= name_position < buffer.size():
                raise ValueError(f"Animation name offset {name_offset} is outside the MDL")
            with buffer.read_from_offset(name_position):
                name = buffer.read_ascii_string()

        if anim_offset and entry_offset + anim_offset >= buffer.size():
            raise ValueError(f"Animation data offset {anim_offset} is outside the MDL")

        return cls(entry_offset, version, name, fps, AnimDescFlags(flags), frame_count, movement_count,
                   movement_offset, bbox, anim_offset, ikrule_count, ikrule_offset)

    def read_animations(self, buffer: Buffer, bones: list[Bone],
                        ani_buffer: Buffer | None = None,
                        block_table: list | None = None) -> dict[str, npt.NDArray] | None:
        if self.frame_count == 0:
            return {}
        if self.anim_offset == 0:
            raise ValueError(f"Animation {self.name!r} has frames but no animation data offset")

        buffer.seek(self._entry_offset + self.anim_offset)
        if self.version == 39:
            return self._read_mdl_animations(buffer, bones, self.frame_count)
        return self._read_fixed_animations(buffer, bones)

    def _read_fixed_animations(self, buffer: Buffer, bones: list[Bone]) -> dict[str, npt.NDArray]:
        animation_start = buffer.tell()
        record_size = 32
        records_end = animation_start + len(bones) * record_size
        if records_end > buffer.size():
            raise ValueError(
                f"Animation {self.name!r} needs {len(bones)} fixed bone records beyond the end of the MDL"
            )

        is_delta = bool(self.flags & AnimDescFlags.DELTA)
        animation_data: dict[str, npt.NDArray] = {}
        for bone_index, bone in enumerate(bones):
            record_offset = animation_start + bone_index * record_size
            buffer.seek(record_offset)
            flags_value = buffer.read_uint32()
            if flags_value & ~int(LegacyAnimBoneFlags.POSITION | LegacyAnimBoneFlags.ROTATION):
                raise ValueError(
                    f"Animation {self.name!r} bone {bone_index} has unsupported flags 0x{flags_value:08X}"
                )
            flags = LegacyAnimBoneFlags(flags_value)

            position_offsets = None
            raw_position = None
            if flags & LegacyAnimBoneFlags.POSITION:
                position_offsets = buffer.read_fmt("3i")
                self._validate_channel_offsets(position_offsets, record_offset, buffer)
            else:
                raw_position = buffer.read_fmt("3f")

            rotation_offsets = None
            raw_rotation = None
            if flags & LegacyAnimBoneFlags.ROTATION:
                rotation_offsets = buffer.read_fmt("3i")
                self._validate_channel_offsets(rotation_offsets, record_offset, buffer)
                buffer.skip(4)
            else:
                raw_rotation = buffer.read_fmt("4f")

            track = np.zeros((self.frame_count,), ANIM_DTYPE)
            if position_offsets is None:
                track["pos"] = raw_position
            else:
                base_position = np.zeros(3, np.float32) if is_delta else np.asarray(bone.position, np.float32)
                track["pos"] = self._decode_channels(
                    buffer, record_offset, position_offsets, bone.position_scale, base_position
                )

            if rotation_offsets is None:
                track["rot"] = raw_rotation
            else:
                base_rotation = np.zeros(3, np.float32) if is_delta else np.asarray(bone.rotation, np.float32)
                rotations = self._decode_channels(
                    buffer, record_offset, rotation_offsets, bone.rotation_scale, base_rotation
                )
                track["rot"] = euler_to_quat(rotations)

            animation_data[bone.name] = track

        buffer.seek(records_end)
        return animation_data

    @staticmethod
    def _validate_channel_offsets(offsets: tuple[int, int, int], record_offset: int, buffer: Buffer) -> None:
        for offset in offsets:
            if offset < 0:
                raise ValueError(f"Negative legacy animation channel offset {offset}")
            if offset and record_offset + offset >= buffer.size():
                raise ValueError(f"Legacy animation channel offset {offset} is outside the MDL")

    def _decode_channels(self, buffer: Buffer, record_offset: int, offsets: tuple[int, int, int],
                         scales: Vector3[float], base_values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        values = np.zeros((self.frame_count, 3), np.float32)
        for axis, offset in enumerate(offsets):
            if offset:
                with buffer.read_from_offset(record_offset + offset):
                    values[:, axis] = self._read_mdl_anim_values(buffer, self.frame_count, scales[axis])
        values += base_values
        return values
