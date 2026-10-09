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

"""OpenArm processor and camera contracts; no downloads or GPU needed."""

import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

from gr00t.data.camera_schema import complete_camera_frames
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.data.utils import parse_modality_configs, to_json_serializable
from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing_module
from gr00t.model.gr00t_n1d7.image_augmentations import build_camera_profile_transform
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "openarm_bimanual_rel_config", ROOT / "examples/OpenArm/openarm_bimanual_rel_config.py"
)
openarm_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(openarm_config)


@pytest.mark.parametrize("as_numpy", [False, True])
def test_ordinary_vlm_frames_keep_upstream_pil_conversion(as_numpy):
    processor = object.__new__(Gr00tN1d7Processor)
    processor.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
    images = torch.arange(2 * 3 * 8 * 10, dtype=torch.int64).to(torch.uint8).reshape(2, 3, 8, 10)
    result = processor._apply_vlm_processing(images.numpy() if as_numpy else images, "test")
    frames = result["vlm_content"]["images"]
    for expected, frame in zip(images, frames):
        assert isinstance(frame, Image.Image)
        np.testing.assert_array_equal(np.asarray(frame), expected.permute(1, 2, 0).numpy())


def test_camera_profile_vlm_frames_keep_hwc_layout():
    processor = object.__new__(Gr00tN1d7Processor)
    processor.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
    images = [np.full((8, 10, 3), 17, dtype=np.uint8), np.full((6, 12, 3), 29, dtype=np.uint8)]
    frames = processor._apply_vlm_processing(images, "test")["vlm_content"]["images"]
    for expected, frame in zip(images, frames):
        assert frame is expected


def test_black_camera_slots_and_roundtrip():
    video = ModalityConfig(
        [0],
        ["wrist", "ceiling", "head"],
        optional_camera_shapes={"ceiling": (6, 9), "head": (7, 12)},
    )
    restored = parse_modality_configs(to_json_serializable({"robot": {"video": video}}))["robot"][
        "video"
    ]
    assert list(OmegaConf.structured(restored).optional_camera_shapes["head"]) == [7, 12]
    wrist = np.full((1, 6, 9, 3), 73, dtype=np.uint8)
    result = complete_camera_frames(
        {"wrist": wrist}, restored.modality_keys, restored.optional_camera_shapes
    )
    assert list(result) == video.modality_keys
    assert result["wrist"] is wrist
    assert result["head"].shape == (1, 7, 12, 3)
    assert not result["head"].any()
    with pytest.raises(ValueError, match="No configured camera"):
        complete_camera_frames({}, video.modality_keys, video.optional_camera_shapes)
    with pytest.raises(ValueError, match="Required camera"):
        complete_camera_frames({"head": wrist}, video.modality_keys, video.optional_camera_shapes)


@pytest.mark.parametrize("fraction", [1.0, 0.95])
def test_profile_fast_path_matches_eval(fraction):
    profiles = openarm_config.openarm_cam_v2
    assert profiles["wrist_right"]["crop_xyxy"] == [51, 0, 909, 600]
    processor = object.__new__(Gr00tN1d7Processor)
    rng = np.random.default_rng(42)
    for camera, original in profiles.items():
        cfg = dict(original, source_crop_fraction=fraction)
        height, width = (720, 1280) if camera.startswith("head") else (600, 960)
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        expected = build_camera_profile_transform(cfg, training=False)(image=image)["image"]
        actual = processor._fast_apply_eval_camera_profile_image(image, cfg)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("meta_has_optional", [False, True])
def test_loader_missing_meta_or_feature_is_explicitly_optional(monkeypatch, meta_has_optional):
    loader = object.__new__(LeRobotEpisodeLoader)
    loader.dataset_path = Path("/fake/dataset")
    loader.modality_meta = {"video": {"wrist": {"original_key": "observation.images.wrist"}}}
    if meta_has_optional:
        loader.modality_meta["video"]["head"] = {"original_key": "observation.images.head"}
    loader.feature_config = {"observation.images.wrist": {"dtype": "video"}}
    config = {
        "video": ModalityConfig([0], ["wrist", "head"], optional_camera_shapes={"head": (7, 12)})
    }
    loader.modality_configs = loader._parse_and_validate_modality_configs(config)
    assert loader._video_key_mapping == {}
    assert loader._missing_video_keys == {"head"}
    loader.video_path_pattern = "videos/{video_key}/{episode_index}.mp4"
    loader.chunk_size = 1000
    loader.decoder_kwargs = None
    monkeypatch.setattr(
        "gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices",
        lambda *a, **kw: np.full((2, 6, 9, 3), 9, dtype=np.uint8),
    )
    result = loader._load_video_data(0, np.array([0, 1]))
    assert result["head"].shape == (2, 7, 12, 3)
    assert not result["head"].any()

    def broken_video(*args, **kwargs):
        raise OSError("decoder failed")

    monkeypatch.setattr(
        "gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices", broken_video
    )
    with pytest.raises(OSError, match="decoder failed"):
        loader._load_video_data(0, np.array([0, 1]))


@pytest.mark.parametrize("training", [False, True])
def test_embodiment_default_profile_applies_in_training_and_eval(training):
    processor = object.__new__(Gr00tN1d7Processor)
    processor.training = training
    processor.camera_image_profiles = {"openarm_cam_v2": openarm_config.openarm_cam_v2}
    processor.default_camera_image_profile_by_embodiment = {
        EmbodimentTag.OPENARM_BIMANUAL_REL.value: "openarm_cam_v2"
    }
    name, _ = processor._resolve_camera_image_profile(EmbodimentTag.OPENARM_BIMANUAL_REL)
    assert name == "openarm_cam_v2"
    assert processor._resolve_camera_image_profile(EmbodimentTag.LIBERO_PANDA) == (None, None)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    fixture = ROOT / "tests/fixtures/processor_config"
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        shutil.copyfile(fixture / name, tmp_path / name)
    vlm = MagicMock(return_value={"input_ids": torch.ones(1, 1, dtype=torch.long)})
    vlm.apply_chat_template.return_value = "move"
    monkeypatch.setattr(processing_module, "build_processor", lambda *args, **kwargs: vlm)
    return tmp_path


def test_released_checkpoint_layout_loads(checkpoint):
    # Released OpenArm checkpoints save a 32-step horizon next to longer embodiments,
    # a fixed state history length and no openarm slot in older mappings.
    config_file = checkpoint / "processor_config.json"
    config = json.loads(config_file.read_text())
    config["processor_kwargs"].update(max_action_horizon=8, max_state_history_length=1)
    config_file.write_text(json.dumps(config))
    mapping = json.loads((checkpoint / "embodiment_id.json").read_text())
    mapping.pop(EmbodimentTag.OPENARM_BIMANUAL_REL.value, None)
    (checkpoint / "embodiment_id.json").write_text(json.dumps(mapping))

    processor = Gr00tN1d7Processor.from_pretrained(checkpoint)
    assert processor.max_action_horizon == 8
    assert processor.embodiment_id_mapping[EmbodimentTag.OPENARM_BIMANUAL_REL.value] == 12
    for tag, slot in mapping.items():
        assert processor.embodiment_id_mapping[tag] == slot
