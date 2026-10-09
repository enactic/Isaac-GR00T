# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.camera_schema import register_camera_image_profile
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


_ACTION_CHUNK_LEN = 32  # typical GR00T chunk horizon

openarm_config = {
    # IMPORTANT: these keys must match the dataset's modality.json "video" keys,
    # NOT the real robot camera nicknames.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "wrist_right",
            "wrist_left",
            "ceiling",
            "head_left",
        ],
        optional_camera_shapes={"ceiling": [600, 960], "head_left": [720, 1280]},
    ),
    # State keys must match modality.json "state"
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm",
            "right_gripper",
            "left_arm",
            "left_gripper",
        ],
    ),
    # Action keys must match modality.json "action"
    "action": ModalityConfig(
        delta_indices=list(range(_ACTION_CHUNK_LEN)),
        modality_keys=[
            "right_arm",
            "right_gripper",
            "left_arm",
            "left_gripper",
        ],
        # Relative arms, absolute grippers.
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Keep consistent with SO100 pattern: "annotation.<key>"
    # Your dataset modality.json has annotation key: "human.action.task_description"
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}

# Per-camera fixed crop [x1, y1, x2, y2] then INTER_AREA resize to [height, width].
# source_crop_fraction < 1 adds a random (train) / center (eval) crop; an optional
# per-camera train: {random_rotate_deg: 10, random_rotate_p: 0.5} adds rotation.
openarm_cam_v2 = {
    "wrist_left": {
        "crop_xyxy": [51, 0, 909, 600],
        "source_crop_fraction": 1.0,
        "final_size": [224, 320],
    },
    "wrist_right": {
        "crop_xyxy": [51, 0, 909, 600],
        "source_crop_fraction": 1.0,
        "final_size": [224, 320],
    },
    "ceiling": {
        "crop_xyxy": [8, 0, 951, 600],
        "source_crop_fraction": 1.0,
        "final_size": [224, 352],
    },
    "head_left": {
        "crop_xyxy": [23, 0, 1257, 720],
        "source_crop_fraction": 1.0,
        "final_size": [224, 384],
    },
}

register_modality_config(openarm_config, embodiment_tag=EmbodimentTag.OPENARM_BIMANUAL_REL)
register_camera_image_profile(
    "openarm_cam_v2", openarm_cam_v2, embodiment_tag=EmbodimentTag.OPENARM_BIMANUAL_REL
)
