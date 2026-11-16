from dataclasses import dataclass

from ....shared.vector_types import Vector3
from ....utils import Buffer
from .event import Event


@dataclass(slots=True)
class LegacySequence:
    _entry_offset: int
    version: int
    name: str
    activity_name: str
    flags: int
    activity: int
    activity_weight: int
    events: list[Event]
    bbox: tuple[Vector3[float], Vector3[float]]
    blend_count: int
    anim_desc_indices: list[int]
    frame_count: int | None
    movement_offset: int
    group_size: tuple[int, int]
    param_offset: tuple[int, int]
    param_start: tuple[float, float]
    param_end: tuple[float, float]
    param_parent: int
    sequence_group: int
    fade_in_time: float
    fade_out_time: float
    local_entry_node: int
    local_exit_node: int
    node_flags: int
    entry_phase: float
    exit_phase: float
    last_frame: float
    next_sequence: int
    pose: int
    ikrule_count: int
    auto_layer_count: int
    auto_layer_offset: int
    weight_offset: int
    pose_key_offset: int
    ik_lock_count: int
    ik_lock_offset: int
    key_value_offset: int
    key_value_size: int
    cycle_pose_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        if version == 32:
            return cls._from_v32(buffer)
        if version in (35, 36):
            return cls._from_v35(buffer, version)
        if version == 39:
            return cls._from_v39(buffer)
        raise ValueError(f"Unsupported legacy sequence version {version}")

    @classmethod
    def _from_v32(cls, buffer: Buffer):
        entry_offset = buffer.tell()
        name_offset, activity_name_offset, flags, activity, activity_weight, event_count, event_offset = \
            buffer.read_fmt("7i")
        bbox = buffer.read_fmt("3f"), buffer.read_fmt("3f")
        frame_count, anim_desc_index, _unknown, sequence_group = buffer.read_fmt("4i")
        group_size = buffer.read_fmt("2I")
        param_offset = buffer.read_fmt("2i")
        param_start = buffer.read_fmt("2f")
        param_end = buffer.read_fmt("2f")
        param_parent = buffer.read_int32()
        fade_in_time, fade_out_time = buffer.read_fmt("2f")
        local_entry_node, local_exit_node, node_flags = buffer.read_fmt("3i")
        entry_phase, exit_phase, last_frame = buffer.read_fmt("3f")
        next_sequence, pose = buffer.read_fmt("2i")
        ikrule_count, auto_layer_count, auto_layer_offset, _stored_weight_offset, pose_key_offset = \
            buffer.read_fmt("5i")
        ik_lock_count, ik_lock_offset, key_value_offset, key_value_size = buffer.read_fmt("4i")
        buffer.skip(2 * 4)
        cls._require_record_position(buffer, entry_offset, 188, 4284)

        name = cls._read_relative_string(buffer, entry_offset, name_offset)
        activity_name = cls._read_relative_string(buffer, entry_offset, activity_name_offset)
        events = cls._read_events(buffer, entry_offset, event_count, event_offset, 32)
        weight_offset = event_offset + event_count * 80 if event_offset else 0

        return cls(entry_offset, 32, name, activity_name, flags, activity, activity_weight, events, bbox, 1,
                   [anim_desc_index], frame_count, 0, group_size, param_offset, param_start, param_end, param_parent,
                   sequence_group, fade_in_time, fade_out_time, local_entry_node, local_exit_node, node_flags,
                   entry_phase, exit_phase, last_frame, next_sequence, pose, ikrule_count, auto_layer_count,
                   auto_layer_offset, weight_offset, pose_key_offset, ik_lock_count, ik_lock_offset, key_value_offset,
                   key_value_size, 0)

    @classmethod
    def _from_v35(cls, buffer: Buffer, version: int):
        entry_offset = buffer.tell()
        name_offset, activity_name_offset, flags, activity, activity_weight, event_count, event_offset = \
            buffer.read_fmt("7i")
        bbox = buffer.read_fmt("3f"), buffer.read_fmt("3f")
        blend_count = buffer.read_uint32()
        animation_indices = buffer.read_fmt("256h")
        movement_offset = buffer.read_uint32()
        group_size = buffer.read_fmt("2I")
        param_offset = buffer.read_fmt("2i")
        param_start = buffer.read_fmt("2f")
        param_end = buffer.read_fmt("2f")
        param_parent = buffer.read_int32()
        sequence_group = buffer.read_int32()
        fade_in_time, fade_out_time = buffer.read_fmt("2f")
        local_entry_node, local_exit_node, node_flags = buffer.read_fmt("3i")
        entry_phase, exit_phase, last_frame = buffer.read_fmt("3f")
        next_sequence, pose = buffer.read_fmt("2i")
        ikrule_count, auto_layer_count, auto_layer_offset, _stored_weight_offset, pose_key_offset = \
            buffer.read_fmt("5i")
        ik_lock_count, ik_lock_offset, key_value_offset, key_value_size = buffer.read_fmt("4i")
        buffer.skip(3 * 4)
        cls._require_record_position(buffer, entry_offset, 700, 4284)

        if blend_count > len(animation_indices):
            raise ValueError(f"Legacy sequence declares {blend_count} blends, maximum is {len(animation_indices)}")
        name = cls._read_relative_string(buffer, entry_offset, name_offset)
        activity_name = cls._read_relative_string(buffer, entry_offset, activity_name_offset)
        events = cls._read_events(buffer, entry_offset, event_count, event_offset, version)
        weight_offset = event_offset + event_count * 80 if event_offset else 0

        return cls(entry_offset, version, name, activity_name, flags, activity, activity_weight, events, bbox,
                   blend_count, list(animation_indices[:blend_count]), None, movement_offset, group_size, param_offset,
                   param_start, param_end, param_parent, sequence_group, fade_in_time, fade_out_time,
                   local_entry_node, local_exit_node, node_flags, entry_phase, exit_phase, last_frame, next_sequence,
                   pose, ikrule_count, auto_layer_count, auto_layer_offset, weight_offset, pose_key_offset,
                   ik_lock_count, ik_lock_offset, key_value_offset, key_value_size, 0)

    @classmethod
    def _from_v39(cls, buffer: Buffer):
        entry_offset = buffer.tell()
        name_offset, activity_name_offset, flags, activity, activity_weight, event_count, event_offset = \
            buffer.read_fmt("7i")
        bbox = buffer.read_fmt("3f"), buffer.read_fmt("3f")
        blend_count, anim_index_offset, movement_offset = buffer.read_fmt("3I")
        group_size = buffer.read_fmt("2I")
        param_offset = buffer.read_fmt("2i")
        param_start = buffer.read_fmt("2f")
        param_end = buffer.read_fmt("2f")
        param_parent = buffer.read_int32()
        fade_in_time, fade_out_time = buffer.read_fmt("2f")
        local_entry_node, local_exit_node, node_flags = buffer.read_fmt("3i")
        entry_phase, exit_phase, last_frame = buffer.read_fmt("3f")
        next_sequence, pose = buffer.read_fmt("2i")
        ikrule_count, auto_layer_count, auto_layer_offset, weight_offset, pose_key_offset = buffer.read_fmt("5I")
        ik_lock_count, ik_lock_offset, key_value_offset, key_value_size, cycle_pose_offset = buffer.read_fmt("5I")
        buffer.skip(7 * 4)
        cls._require_record_position(buffer, entry_offset, 208, 208)

        grid_count = group_size[0] * group_size[1]
        if blend_count and grid_count and blend_count != grid_count:
            raise ValueError(
                f"Legacy v39 sequence declares {blend_count} blends for a {group_size[0]}x{group_size[1]} grid"
            )
        index_count = grid_count or blend_count
        anim_desc_indices = []
        if index_count:
            if anim_index_offset == 0:
                raise ValueError("Legacy v39 sequence has blends but no animation index offset")
            cls._validate_relative_offset(buffer, entry_offset, anim_index_offset, index_count * 2,
                                          "animation index")
            with buffer.read_from_offset(entry_offset + anim_index_offset):
                anim_desc_indices = list(buffer.read_fmt(f"{index_count}h"))

        name = cls._read_relative_string(buffer, entry_offset, name_offset)
        activity_name = cls._read_relative_string(buffer, entry_offset, activity_name_offset)
        events = cls._read_events(buffer, entry_offset, event_count, event_offset, 39)

        return cls(entry_offset, 39, name, activity_name, flags, activity, activity_weight, events, bbox, blend_count,
                   anim_desc_indices, None, movement_offset, group_size, param_offset, param_start, param_end,
                   param_parent, 0, fade_in_time, fade_out_time, local_entry_node, local_exit_node, node_flags,
                   entry_phase, exit_phase, last_frame, next_sequence, pose, ikrule_count, auto_layer_count,
                   auto_layer_offset, weight_offset, pose_key_offset, ik_lock_count, ik_lock_offset, key_value_offset,
                   key_value_size, cycle_pose_offset)

    def get_bone_weights(self, buffer: Buffer, bone_count: int) -> list[float]:
        if bone_count == 0:
            return []
        if self.weight_offset == 0:
            raise ValueError(f"Sequence {self.name!r} has no bone-weight offset")
        self._validate_relative_offset(buffer, self._entry_offset, self.weight_offset, bone_count * 4, "bone weight")
        with buffer.read_from_offset(self._entry_offset + self.weight_offset):
            return list(buffer.read_fmt(f"{bone_count}f"))

    @staticmethod
    def _require_record_position(buffer: Buffer, entry_offset: int, parsed_size: int, record_size: int) -> None:
        expected = entry_offset + parsed_size
        if buffer.tell() != expected:
            raise ValueError(f"Legacy sequence parser consumed {buffer.tell() - entry_offset}, expected {parsed_size}")
        if entry_offset + record_size > buffer.size():
            raise ValueError(f"Legacy sequence record extends beyond the end of the MDL")
        buffer.seek(entry_offset + record_size)

    @staticmethod
    def _validate_relative_offset(buffer: Buffer, entry_offset: int, offset: int, size: int, label: str) -> None:
        start = entry_offset + offset
        if offset <= 0 or start < 0 or start + size > buffer.size():
            raise ValueError(f"Legacy sequence {label} offset {offset} is outside the MDL")

    @classmethod
    def _read_relative_string(cls, buffer: Buffer, entry_offset: int, offset: int) -> str:
        if offset <= 0:
            return ""
        cls._validate_relative_offset(buffer, entry_offset, offset, 1, "string")
        with buffer.read_from_offset(entry_offset + offset):
            return buffer.read_ascii_string()

    @classmethod
    def _read_events(cls, buffer: Buffer, entry_offset: int, count: int, offset: int,
                     version: int) -> list[Event]:
        if count == 0:
            return []
        cls._validate_relative_offset(buffer, entry_offset, offset, count * 80, "event")
        events = []
        with buffer.read_from_offset(entry_offset + offset):
            for _ in range(count):
                events.append(Event.from_buffer(buffer, version))
        return events
