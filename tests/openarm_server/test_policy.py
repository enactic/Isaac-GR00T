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

"""OpenArm policy server contracts; no checkpoint or GPU required."""

import json
from types import SimpleNamespace

from gr00t.data.types import ModalityConfig
import numpy as np
from openarm_server import policy as policy_module
from openarm_server.policy import OpenArmPolicy, read_qpos, resolve_checkpoint
import pyarrow as pa


JOINTS = (("right_arm", 7), ("right_gripper", 1), ("left_arm", 7), ("left_gripper", 1))
CAMERA_METADATA = {
    f"camera_{key}.{dim}": value
    for key in ("wrist_right", "wrist_left")
    for dim, value in (("height", 2), ("width", 3))
}


def fake_gr00t_policy(horizon=4):
    calls = []

    def get_action(model_input, options=None):
        calls.append((model_input, options))
        chunk = {
            key: np.full((1, horizon, width), value, dtype=np.float32)
            + np.arange(horizon, dtype=np.float32)[None, :, None] * 10
            for value, (key, width) in enumerate(JOINTS)
        }
        chunk["state_right_arm"] = np.full((1, horizon, 7), 999, dtype=np.float32)
        return chunk, {}

    return SimpleNamespace(
        modality_configs={
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=["wrist_right", "wrist_left", "ceiling", "head_left"],
                optional_camera_shapes={"ceiling": [2, 3], "head_left": [3, 4]},
            ),
            "language": ModalityConfig([0], ["annotation.human.action.task_description"]),
        },
        get_action=get_action,
        reset=lambda: {},
        calls=calls,
    )


def write_observation(path, cameras=("camera_wrist_right", "camera_wrist_left")):
    """Write one dora-openarm-observer row with simulation list<float32> positions."""
    columns = {"position": pa.array([np.arange(16, dtype=np.float32)], pa.list_(pa.float32()))}
    for index, name in enumerate(cameras):
        columns[name] = pa.array([[10 * (index + 1)] * 18], pa.list_(pa.uint8()))
    columns["task_prompt"] = pa.array(["pillow"])
    batch = pa.record_batch(list(columns.values()), names=list(columns))
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, batch.schema) as writer:
        writer.write(batch)
    return str(path)


class Lines:
    def __init__(self, requests):
        self.lines = [json.dumps(request) + "\n" for request in requests]
        self.responses = []

    def __iter__(self):
        return iter(self.lines)

    def write(self, text):
        self.responses.append(json.loads(text))

    def flush(self):
        pass


def test_serve_returns_full_chunks_and_rejects_bad_requests(tmp_path):
    gr00t = fake_gr00t_policy()
    request = {
        "name": "inference",
        "data_path": write_observation(tmp_path / "obs.arrow"),
        "metadata": CAMERA_METADATA,
    }
    missing_wrist = write_observation(tmp_path / "missing.arrow", cameras=("camera_wrist_right",))
    io = Lines([{**request, "reset": True}, request, {**request, "data_path": missing_wrist}])
    OpenArmPolicy(gr00t, denoising_steps=4).serve(io)

    for response in io.responses[:2]:
        positions = np.asarray(response["positions"])
        assert positions.shape == (4, 16)
        np.testing.assert_array_equal(positions[0], [0] * 7 + [1] + [2] * 7 + [3])
        np.testing.assert_array_equal(positions[:, 0], [0, 10, 20, 30])
        assert response["interval"] == 33_333_333
    assert io.responses[2]["positions"] == [] and "wrist_left" in io.responses[2]["error"]

    model_input, options = gr00t.calls[0]
    assert options == {"denoising_steps": 4}
    assert list(model_input["video"]) == ["wrist_right", "wrist_left", "ceiling", "head_left"]
    assert model_input["video"]["wrist_left"].shape == (1, 1, 2, 3, 3)
    assert model_input["video"]["wrist_left"].max() == 20
    assert model_input["video"]["ceiling"].shape == (1, 1, 2, 3, 3)
    assert not model_input["video"]["ceiling"].any()
    np.testing.assert_array_equal(model_input["state"]["right_arm"][0, 0], np.arange(7))
    np.testing.assert_array_equal(model_input["state"]["left_gripper"][0, 0], [15])
    assert model_input["language"] == {"annotation.human.action.task_description": [["pillow"]]}


def test_real_arm_qpos_structs_are_flattened():
    # dora-openarm sends struct<qpos: list<float32>> per arm; the observer concatenates
    # right then left into one list<struct<...>> row.
    qpos_type = pa.struct([("qpos", pa.list_(pa.float32()))])
    qpos = np.arange(16, dtype=np.float32)
    position = pa.array([[{"qpos": qpos[:8]}, {"qpos": qpos[8:]}]], pa.list_(qpos_type))[0]
    np.testing.assert_array_equal(read_qpos(position), qpos)


def test_warmup_runs_synthetic_inputs():
    gr00t = fake_gr00t_policy()
    OpenArmPolicy(gr00t).warmup(2, image_height=2, image_width=3)
    assert len(gr00t.calls) == 2
    assert gr00t.calls[0][0]["video"]["wrist_left"].shape == (1, 1, 2, 3, 3)
    assert gr00t.calls[0][0]["state"]["right_arm"].shape == (1, 1, 7)


def test_checkpoint_resolution_downloads_only_inference_files(tmp_path, monkeypatch):
    assert resolve_checkpoint(tmp_path) == tmp_path
    calls = []

    def snapshot_download(repo_id, allow_patterns):
        calls.append((repo_id, allow_patterns))
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert resolve_checkpoint("enactic/GR00T-N1.7-openarm-2-cell-pillow") == tmp_path
    assert calls == [
        ("enactic/GR00T-N1.7-openarm-2-cell-pillow", list(policy_module.INFERENCE_FILE_PATTERNS))
    ]
