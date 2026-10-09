# OpenArm Policy Server

Serves a GR00T N1.7 OpenArm checkpoint to
[`dora-openarm-local-policy-server`](https://github.com/enactic/dora-openarm-local-policy-server)
over a UNIX socket. The server receives observations and returns joint-target
action chunks. Robot control, interpolation and recording stay in the Dora graph.

## Model

[`enactic/GR00T-N1.7-openarm-2-cell-pillow`](https://huggingface.co/enactic/GR00T-N1.7-openarm-2-cell-pillow)
uses the `OPENARM_BIMANUAL_REL` embodiment tag and the `openarm_cam_v2` camera
profile saved in the checkpoint. Checkpoints trained with the
[OpenArm training guide](../examples/OpenArm/README.md) are served the same way. `--checkpoint-dir` accepts either a local
directory or a Hub repo ID. A repo ID downloads only the `*.json` and
`*.safetensors` files. The `nvidia/Cosmos-Reason2-2B` backbone is a gated Hub model.

Built on NVIDIA Cosmos. The model is derived from `nvidia/GR00T-N1.7-3B` and
`nvidia/Cosmos-Reason2-2B`, both under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Start

Install the repository environment (see the [installation guide](../README.md#installation)),
then run from the repository root:

```bash
uv run python openarm_server/policy_server.py \
  --checkpoint-dir enactic/GR00T-N1.7-openarm-2-cell-pillow \
  --socket-path /dev/shm/policy-server.socket \
  --denoising-steps 6
```

Start the server before the Dora dataflow and point the bridge's `SOCKET` at the
same path; the socket is created after the model loads and warms up, and an
existing file at that path is replaced. Each request names an Arrow IPC file
holding one `dora-openarm-observer` observation. `CHECKPOINT_DIR` and
`SOCKET_PATH` set the defaults of `--checkpoint-dir` and `--socket-path`.

Run `python openarm_server/policy_server.py --help` for the full option list.
Inference runs in PyTorch. The scripts in `scripts/deployment`, including the
TensorRT and standalone inference tools, do not apply OpenArm camera profiles
and are not supported for OpenArm checkpoints in this release.

## Actions

The server is stateless: each request returns the full predicted chunk (32
steps at 30 Hz) of 16D absolute joint targets in this order: right arm (7),
right gripper (1), left arm (7), left gripper (1). Action windows, execution resets and the inference rate are
the bridge's job; the server does not slice chunks or wait between requests.

The optional `ceiling` and `head_left` cameras are black-filled when absent. A
missing wrist camera returns an `error` and no actions.

