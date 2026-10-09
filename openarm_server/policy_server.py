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

"""Serve a GR00T N1.7 OpenArm checkpoint to dora-openarm-local-policy-server.

It listens on a Unix domain socket and serves one connected bridge at a time.
"""

import argparse
import logging
import os
import pathlib
import socket
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from openarm_server.policy import OpenArmPolicy, load_gr00t_policy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--checkpoint-dir",
        default=os.getenv("CHECKPOINT_DIR"),
        required=os.getenv("CHECKPOINT_DIR") is None,
        help="Local checkpoint directory or Hugging Face repo ID (default: $CHECKPOINT_DIR).",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="OPENARM_BIMANUAL_REL",
        help="Embodiment tag saved in the checkpoint.",
    )
    parser.add_argument(
        "--socket-path",
        default=os.getenv("SOCKET_PATH", "/dev/shm/policy-server.socket"),
        help="UNIX socket shared with the bridge (default: $SOCKET_PATH).",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for the model.")
    parser.add_argument(
        "--denoising-steps",
        type=int,
        help="Flow-matching denoising steps. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--prompt", default="", help="Instruction used when an observation has no task_prompt."
    )
    parser.add_argument(
        "--image-preprocess-workers",
        default=4,
        type=int,
        help="Threads for per-camera crop/resize preprocessing.",
    )
    parser.add_argument(
        "--warmup-steps", default=1, type=int, help="Synthetic inferences before serving."
    )
    parser.add_argument("--warmup-image-height", default=480, type=int)
    parser.add_argument("--warmup-image-width", default=640, type=int)
    args = parser.parse_args(argv)
    if args.image_preprocess_workers <= 0:
        parser.error("--image-preprocess-workers must be > 0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gr00t_policy = load_gr00t_policy(
        args.checkpoint_dir,
        args.embodiment_tag,
        device=args.device,
        image_preprocess_workers=args.image_preprocess_workers,
    )
    policy = OpenArmPolicy(gr00t_policy, denoising_steps=args.denoising_steps, prompt=args.prompt)
    policy.warmup(
        args.warmup_steps,
        image_height=args.warmup_image_height,
        image_width=args.warmup_image_width,
    )

    socket_path = args.socket_path
    if os.path.exists(socket_path):
        os.remove(socket_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(socket_path)
            sock.listen()
            logging.info("Listening on %s", socket_path)
            while True:
                conn, _ = sock.accept()
                logging.info("Connected")
                try:
                    with conn, conn.makefile("rw") as io:
                        policy.serve(io)
                except OSError:
                    logging.exception("Policy client disconnected")
    finally:
        gr00t_policy.processor.close_image_preprocess_workers()
        if os.path.exists(socket_path):
            os.remove(socket_path)


if __name__ == "__main__":
    main()
