"""Object-free, bit-exact action decoding for the default N1.7 embodiment."""

from __future__ import annotations

import numpy as np


OXE_DROID = "oxe_droid_relative_eef_relative_joint"


class FastOXEDecoder:
    """Remove Pose/ActionChunk objects without changing floating-point order."""

    def __init__(self, processor) -> None:
        from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType

        self._processor = processor
        self._fallback = processor.decode_action
        self._sap = processor.state_action_processor
        config = processor.modality_configs[OXE_DROID]["action"]
        expected = (
            (
                ActionRepresentation.RELATIVE,
                ActionType.EEF,
                ActionFormat.XYZ_ROT6D,
                "eef_9d",
            ),
            (
                ActionRepresentation.ABSOLUTE,
                ActionType.NON_EEF,
                ActionFormat.DEFAULT,
                "gripper_position",
            ),
            (
                ActionRepresentation.RELATIVE,
                ActionType.NON_EEF,
                ActionFormat.DEFAULT,
                "joint_position",
            ),
        )
        actual = tuple(
            (item.rep, item.type, item.format, item.state_key)
            for item in config.action_configs
        )
        self.supported = tuple(config.modality_keys) == (
            "eef_9d",
            "gripper_position",
            "joint_position",
        ) and actual == expected

    def __call__(self, action, embodiment_tag, state=None):
        if (
            embodiment_tag.value != OXE_DROID
            or not self.supported
            or state is None
            or np.asarray(action).ndim != 3
        ):
            return self._fallback(action, embodiment_tag, state)

        from gr00t.data.utils import unnormalize_values_meanstd, unnormalize_values_minmax

        config = self._processor.modality_configs[OXE_DROID]["action"]
        horizon = len(config.delta_indices)
        mean_std = set(config.mean_std_embedding_keys or ())
        values = {}
        offset = 0
        for key in config.modality_keys:
            params = self._sap.norm_params[OXE_DROID]["action"][key]
            width = int(params["dim"].item())
            normalized = action[..., :horizon, offset : offset + width]
            decode = unnormalize_values_meanstd if key in mean_std else unnormalize_values_minmax
            values[key] = decode(normalized, params)
            offset += width

        if self._sap.use_relative_action:
            values["eef_9d"] = _absolute_xyz_rot6d_exact(
                values["eef_9d"], state["eef_9d"]
            )
            values["joint_position"] = _absolute_joint_exact(
                values["joint_position"], state["joint_position"]
            )
        return values


def _xyz_rot6d_homogeneous(value: np.ndarray) -> np.ndarray:
    """Match ``EndEffectorPose(..., rotation_type="rot6d").homogeneous``."""

    from gr00t.data.state_action.pose import EndEffectorPose
    from scipy.spatial.transform import Rotation

    translation = np.array(value[:3])
    rotation = np.array(value[3:])
    rotation_matrix = EndEffectorPose._rot6d_to_matrix(rotation)
    scipy_rotation = Rotation.from_matrix(rotation_matrix)
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = scipy_rotation.as_matrix()
    homogeneous[:3, 3] = translation
    return homogeneous


def _absolute_xyz_rot6d_exact(relative: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compose EEF actions with exact math and batched SciPy conversions."""

    from gr00t.data.state_action.pose import EndEffectorPose
    from scipy.spatial.transform import Rotation

    relative = np.asarray(relative)
    reference = np.asarray(reference)
    batched = relative.ndim == 3
    if not batched:
        relative = relative[None]
    if reference.ndim == 2:
        reference = reference[None]

    absolute_batches = []
    for relative_chunk, reference_chunk in zip(relative, reference):
        reference_homogeneous = _xyz_rot6d_homogeneous(reference_chunk[-1])
        relative_rotation_matrices = np.stack(
            [
                EndEffectorPose._rot6d_to_matrix(np.array(pose[3:]))
                for pose in relative_chunk
            ]
        )
        relative_rotations = Rotation.from_matrix(relative_rotation_matrices).as_matrix()
        relative_homogeneous = np.repeat(
            np.eye(4)[None, :, :], len(relative_chunk), axis=0
        )
        relative_homogeneous[:, :3, :3] = relative_rotations
        relative_homogeneous[:, :3, 3] = relative_chunk[:, :3]

        # Keep one 4x4 matmul per timestep. Splitting this into a batched 3x3
        # rotation and translation changes the final ULP on some inputs.
        absolute_homogeneous = np.stack(
            [reference_homogeneous @ pose for pose in relative_homogeneous]
        )
        absolute_rotations = Rotation.from_matrix(
            absolute_homogeneous[:, :3, :3]
        ).as_matrix()
        absolute_batches.append(
            np.concatenate(
                (
                    absolute_homogeneous[:, :3, 3],
                    absolute_rotations[:, :2, :].reshape(-1, 6),
                ),
                axis=1,
            )
        )
    output = np.stack(absolute_batches, axis=0)
    return output if batched else output[0]


def _absolute_joint_exact(relative: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match JointPose conversion and per-pose addition used by upstream."""

    relative = np.asarray(relative)
    reference = np.asarray(reference)
    batched = relative.ndim == 3
    if not batched:
        relative = relative[None]
    if reference.ndim == 2:
        reference = reference[None]

    absolute_batches = []
    for relative_chunk, reference_chunk in zip(relative, reference):
        reference_joints = np.array(reference_chunk[-1], dtype=np.float64)
        absolute_chunk = []
        for relative_pose in relative_chunk:
            relative_joints = np.array(relative_pose, dtype=np.float64)
            absolute_joints = reference_joints + relative_joints
            absolute_chunk.append(np.array(absolute_joints, dtype=np.float64))
        absolute_batches.append(np.array(absolute_chunk))
    output = np.stack(absolute_batches, axis=0)
    return output if batched else output[0]


def install_fast_decode(policy) -> FastOXEDecoder | None:
    processor = policy.processor
    current = getattr(processor, "_instinctflash_fast_decoder", None)
    if current is not None:
        return current
    decoder = FastOXEDecoder(processor)
    if not decoder.supported:
        return None
    processor.decode_action = decoder
    processor._instinctflash_fast_decoder = decoder
    return decoder


__all__ = ["FastOXEDecoder", "install_fast_decode"]
