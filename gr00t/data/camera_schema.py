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

"""Camera image profiles and optional camera slots for image preprocessing."""

import numpy as np


# Named per-camera crop/resize profiles, saved in the processor config.
CAMERA_IMAGE_PROFILES: dict[str, dict] = {}
# Embodiment tag value -> profile name used when a datapoint selects none.
DEFAULT_CAMERA_IMAGE_PROFILES: dict[str, str] = {}


def register_camera_image_profile(name, profile, embodiment_tag=None):
    """Register a profile, optionally as the default for an embodiment tag."""
    CAMERA_IMAGE_PROFILES[name] = profile
    if embodiment_tag is not None:
        DEFAULT_CAMERA_IMAGE_PROFILES[getattr(embodiment_tag, "value", embodiment_tag)] = name


def complete_camera_frames(images, camera_keys, optional_camera_shapes=None):
    """Preserve camera order and THWC layout; never hide a failed video decode."""
    optional_camera_shapes = optional_camera_shapes or {}
    available = [images[key] for key in camera_keys if key in images]
    if not available:
        raise ValueError("No configured camera is available")
    frame_count = len(available[0])
    completed = {}
    for key in camera_keys:
        if key in images:
            frames = np.asarray(images[key])
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"Camera {key!r} must be uint8 THWC RGB")
            if len(frames) != frame_count:
                raise ValueError(f"Camera {key!r} has a different history length")
            completed[key] = frames
        elif key in optional_camera_shapes:
            height, width = optional_camera_shapes[key]
            completed[key] = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Required camera {key!r} is missing")
    return completed
