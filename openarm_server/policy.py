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

"""GR00T N1.7 OpenArm policy served to dora-openarm-local-policy-server.

Each JSON-line request names an Arrow IPC file holding one dora-openarm-observer
observation; each response returns the full predicted chunk of 16D absolute
joint targets. Action windows, execution resets and rate limiting are left to
the bridge.
"""

import json
import logging
from pathlib import Path
import time

from gr00t.data.camera_schema import complete_camera_frames
import numpy as np
import pyarrow as pa


LOGGER = logging.getLogger(__name__)

# Files needed for inference; training checkpoints may also hold optimizer states.
INFERENCE_FILE_PATTERNS = ("*.json", "*.safetensors")

# OpenArm qpos/action layout: name -> columns of the 16D vector.
JOINT_SLICES = {
    "right_arm": slice(0, 7),
    "right_gripper": slice(7, 8),
    "left_arm": slice(8, 15),
    "left_gripper": slice(15, 16),
}
STATE_DIM = 16
INTERVAL_NS = 33_333_333  # 30 Hz action points


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Return a local checkpoint directory, downloading a Hugging Face repo ID if needed."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    LOGGER.info("Downloading %s from the Hugging Face Hub", checkpoint)
    return Path(snapshot_download(str(checkpoint), allow_patterns=list(INFERENCE_FILE_PATTERNS)))


def load_gr00t_policy(checkpoint, embodiment_tag, *, device="cuda:0", image_preprocess_workers=1):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
        model_path=str(resolve_checkpoint(checkpoint)),
        device=device,
        strict=True,
    )
    policy.processor.set_image_preprocess_workers(image_preprocess_workers)
    return policy


def read_qpos(position) -> np.ndarray:
    """Flatten `position` into float32 [16].

    Simulation sends list<float32>; real arms send one struct<qpos: list<float32>>
    per arm.
    """
    values = []
    for item in position.as_py():
        values.extend(next(iter(item.values())) if isinstance(item, dict) else [item])
    qpos = np.asarray(values, dtype=np.float32)
    if qpos.shape != (STATE_DIM,):
        raise ValueError(f"Expected {STATE_DIM}-DoF bimanual qpos, got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("Non-finite input position")
    return qpos


def read_image(observation, metadata, field) -> np.ndarray:
    """Reshape a flat RGB camera field to uint8 [H, W, 3] using the observer metadata."""
    shape = (int(metadata[f"{field}.height"]), int(metadata[f"{field}.width"]), 3)
    return observation[field].values.to_numpy(zero_copy_only=False).reshape(shape)


class OpenArmPolicy:
    """Adapter between the OpenArm wire format and `Gr00tPolicy`.

    Camera and language keys come from the checkpoint's saved modality config.
    Missing optional cameras are black-filled; a missing required camera is an
    error, never padded.
    """

    def __init__(self, policy, *, denoising_steps: int | None = None, prompt: str = ""):
        self.policy = policy
        self.options = {"denoising_steps": denoising_steps} if denoising_steps else None
        self.prompt = prompt
        self.video_config = policy.modality_configs["video"]
        self.language_key = policy.modality_configs["language"].modality_keys[0]

    def model_input(self, frames: dict[str, np.ndarray], qpos: np.ndarray, prompt: str):
        """Build the batched Gr00tPolicy observation (B=1, T=1)."""
        keys = self.video_config.modality_keys
        video = {key: frames[key][np.newaxis] for key in keys if key in frames}
        video = complete_camera_frames(video, keys, self.video_config.optional_camera_shapes)
        return {
            "video": {key: value[np.newaxis] for key, value in video.items()},
            "state": {
                key: qpos[np.newaxis, np.newaxis, cols] for key, cols in JOINT_SLICES.items()
            },
            "language": {self.language_key: [[prompt]]},
        }

    def predict(self, frames, qpos, prompt) -> np.ndarray:
        """Return the full decoded chunk as absolute joint targets [horizon, 16]."""
        chunk, _ = self.policy.get_action(self.model_input(frames, qpos, prompt), self.options)
        horizon = next(iter(chunk.values())).shape[1]
        positions = np.tile(qpos, (horizon, 1))
        for key, cols in JOINT_SLICES.items():
            if key in chunk:
                positions[:, cols] = chunk[key][0]
        return positions

    def infer(self, observation, metadata) -> dict:
        """Predict an action chunk for one Arrow observation row."""
        names = {item.name for item in observation.type}
        frames = {
            key: read_image(observation, metadata, f"camera_{key}")
            for key in self.video_config.modality_keys
            if f"camera_{key}" in names
        }
        prompt = observation["task_prompt"].as_py() if "task_prompt" in names else None
        positions = self.predict(frames, read_qpos(observation["position"]), prompt or self.prompt)
        return {"interval": INTERVAL_NS, "positions": positions.tolist()}

    def warmup(self, steps=1, *, image_height=480, image_width=640):
        """Run synthetic inferences so CUDA initialization does not delay the first request."""
        frames = {
            key: np.zeros((image_height, image_width, 3), dtype=np.uint8)
            for key in self.video_config.modality_keys
        }
        for index in range(steps):
            started = time.perf_counter()
            self.predict(frames, np.zeros(STATE_DIM, dtype=np.float32), self.prompt)
            LOGGER.info(
                "Warmup %d/%d: %.1f ms", index + 1, steps, (time.perf_counter() - started) * 1000
            )

    def serve(self, io):
        """Answer JSON-line requests read from `io` until it is closed."""
        for line in io:
            try:
                request = json.loads(line)
                if request.get("reset"):
                    self.policy.reset()
                with pa.OSFile(request["data_path"], "rb") as f, pa.ipc.open_file(f) as reader:
                    observation = reader.get_batch(0).to_struct_array()[0]
                response = self.infer(observation, request.get("metadata", {}))
                encoded = json.dumps(response, allow_nan=False)
            except Exception as exc:
                LOGGER.exception("Request rejected; no actions returned")
                encoded = json.dumps({"positions": [], "error": str(exc)})
            io.write(encoded + "\n")
            io.flush()
