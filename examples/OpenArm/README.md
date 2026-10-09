# OpenArm N1.7 Training

Fine-tune N1.7 on OpenArm relative-joint data with the shared fine-tuning
entrypoint. 

## Environment

Follow the official [installation guide](../../README.md#installation), then
activate the prepared environment. The root dependency set is unchanged:
Python 3.12, Torch 2.9.0 and Transformers 4.57.3. Use the official platform-specific
environment on aarch64. The [official Dockerfile](../../docker/Dockerfile) is optional.

Run commands from the repository root with the active `python`/`torchrun`.

## Data

Convert an OpenArm recording with the `gr00t` format of
[`openarm_dataset`](https://github.com/enactic/openarm_dataset):

```bash
python -m openarm_dataset.convert --format gr00t /path/to/openarm_recording /path/to/pillow
```

This writes a LeRobot v2.1 dataset with `meta/modality.json`; pass its directory
as `--dataset-path`. Keep the default `--state qpos`. The modality config
[`openarm_bimanual_rel_config.py`](openarm_bimanual_rel_config.py) reads the
`right_arm`, `right_gripper`, `left_arm` and `left_gripper` slices and the
`wrist_right`, `wrist_left`, `ceiling` and `head_left` cameras. Other recorded
slices and cameras, such as `lifter` or `head_right`, are ignored. The `ceiling`
and `head_left` cameras are optional: a dataset without them gets black frames;
the wrist cameras are required.

To train on several datasets, join their directories with `:` in `--dataset-path`.

## Embodiment And Cameras

The `OPENARM_BIMANUAL_REL` tag (projector slot 12) outputs 16D joint actions:
relative arms and absolute grippers, with a 32-step action chunk. Pass
`--action-horizon 32` so the model horizon matches the chunk, as in the released
checkpoints.

Loading the modality config also registers the `openarm_cam_v2` camera profile
as the tag's default. It applies a fixed per-camera crop and an `INTER_AREA`
resize in both training and inference, and is saved in `processor_config.json`
so inference uses the same preprocessing. Edit `openarm_cam_v2` in the config
file to change it: `source_crop_fraction` below 1.0 adds random training crops
and centered inference crops of the same size, and a per-camera
`train: {random_rotate_deg: 10, random_rotate_p: 0.5}` adds training-only rotation.

## Local Training

Example single-GPU warmup (choose a free GPU first):

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag OPENARM_BIMANUAL_REL \
  --modality-config-path examples/OpenArm/openarm_bimanual_rel_config.py \
  --action-horizon 32 \
  --num-gpus 1 --global-batch-size 2 --max-steps 10 \
  --learning-rate 1e-4 --output-dir outputs/smoke
```

For multiple GPUs, use the same entrypoint through torchrun:

```bash
torchrun --standalone --nproc_per_node=4 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag OPENARM_BIMANUAL_REL \
  --modality-config-path examples/OpenArm/openarm_bimanual_rel_config.py \
  --action-horizon 32 \
  --num-gpus 4 --global-batch-size 8 \
  --max-steps 10 --output-dir outputs/train_4gpu
```

Global batch size must be divisible by GPU count; gradient accumulation further
multiplies the optimizer-step batch. Choose GPUs and batch sizes for your hardware.

To resume an interrupted run, repeat the command with the same `--output-dir`
and `--resume-from-checkpoint`; training continues from the latest checkpoint in
that directory with the same settings and GPU count. Starting from another
checkpoint with `--base-model-path` and a new output directory is a new stage:
weights are loaded, while the optimizer, scheduler and step counter start fresh.
Add `--resume-base-ckpt-stats` to keep that checkpoint's normalization statistics.
All options are listed by `python gr00t/experiment/launch_finetune.py --help`.

## Fine-Tuning A Released Checkpoint

The released checkpoint can initialize a new stage directly from the Hub. For
example, to continue training
[`enactic/GR00T-N1.7-openarm-2-cell-pillow`](https://huggingface.co/enactic/GR00T-N1.7-openarm-2-cell-pillow)
on your own pillow data:

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path enactic/GR00T-N1.7-openarm-2-cell-pillow \
  --dataset-path /path/to/pillow \
  --embodiment-tag OPENARM_BIMANUAL_REL \
  --modality-config-path examples/OpenArm/openarm_bimanual_rel_config.py \
  --action-horizon 32 --tune-visual --resume-base-ckpt-stats \
  --num-gpus 1 --global-batch-size 8 --max-steps 1000 \
  --learning-rate 3e-5 --output-dir outputs/pillow_continued
```

The weights and embodiment mapping come from the checkpoint; the optimizer and
schedule start fresh. `--resume-base-ckpt-stats` keeps the released
normalization statistics, which suits more data for the same task; omit it to
recompute statistics from your data.
