# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
import os
import pickle

import cv2
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RobotwinDataPacker(BaseLmdbManipulationDataPacker):
    def __init__(
        self,
        input_path,
        output_path,
        seed_dir,
        task_names=None,
        embodiment=None,
        simulation=True,
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        self.seed_dir = seed_dir
        self.task_names = task_names
        self.embodiment = embodiment
        self.simulation = simulation
        self.episodes = self.input_path_handler(self.input_path)

    def _check_valid(self, input_path, task_dir):
        if not os.path.isdir(os.path.join(input_path, task_dir)):
            return None, None, None

        if task_dir.endswith("_pkl"):
            camera_name = task_dir.split("_")[-2]
            task_name = task_dir.replace(f"_{camera_name}_pkl", "")
        else:
            camera_name = task_dir.split("_")[-1]
            task_name = task_dir.replace(f"_{camera_name}", "")

        valid_seed_file = None
        for seed_file in os.listdir(self.seed_dir):
            if self.embodiment is not None:
                if seed_file == f"{task_name}_{self.embodiment}.txt":
                    valid_seed_file = seed_file
                    break
            elif seed_file == f"{task_name}.txt":
                valid_seed_file = seed_file
                break
        return camera_name, task_name, valid_seed_file

    def input_path_handler(self, input_path):
        episodes = []
        for task_dir in os.listdir(input_path):
            camera_name, task_name, valid_seed_file = self._check_valid(
                input_path, task_dir
            )
            if valid_seed_file is None:
                logger.warning(f"invalid task dir: {task_dir}")
                continue

            seeds = (
                open(os.path.join(self.seed_dir, valid_seed_file), "r")
                .read()
                .strip()
                .split(" ")
            )
            if (
                self.task_names is not None
                and task_name not in self.task_names
            ):
                continue
            for ep in os.listdir(os.path.join(input_path, task_dir)):
                ep_path = os.path.join(input_path, task_dir, ep)
                if not (ep.startswith("episode") and os.path.isdir(ep_path)):
                    continue
                ep_id = int(ep.replace("episode", ""))
                seed = seeds[ep_id]
                episodes.append([task_name, camera_name, ep_path, seed])
        episodes.sort(key=lambda x: (x[0], x[1], int(x[3])))  # sort by seed
        logger.info(f"number of valid episodes: {len(episodes)}")
        return episodes

    def _pack(self):
        num_valid_ep = 0
        for ep_id, ep in enumerate(self.episodes):
            task_name, camera_type, ep_path, seed = ep
            uuid = f"{task_name}_{camera_type}_seed{seed}"
            logger.info(
                f"start process [{ep_id + 1}/{len(self.episodes)}] {uuid}"
            )
            num_steps = 0
            joint_positions = []
            cartesian_positions = []
            extrinsics = {}
            intrinsics = {}
            rgbs = {}
            depths = {}

            frame_data_files = [
                x for x in os.listdir(ep_path) if x.endswith(".pkl")
            ]
            frame_data_files.sort(key=lambda x: int(x[:-4]))
            for pkl in frame_data_files:
                try:
                    pkl_file = os.path.join(ep_path, pkl)
                    frame_data = pickle.load(open(pkl_file, "rb"))
                except Exception:
                    logger.warning(f"invalid pkl file: {pkl_file}")
                    continue

                camera_names = []
                for camera, obs in frame_data["observation"].items():
                    camera_names.append(camera)

                    rgb = obs["rgb"]
                    assert rgb.shape[-1] == 3 and len(rgb.shape) == 3
                    ret, rgb = cv2.imencode(".JPEG", rgb)
                    assert ret

                    depth = obs["depth"]
                    assert len(depth.shape) == 2
                    depth = depth.astype(np.uint16)
                    ret, depth = cv2.imencode(".PNG", depth)
                    assert ret

                    if camera not in intrinsics:
                        intrinsics[camera] = obs["intrinsic_cv"]
                        extrinsics[camera] = []
                        rgbs[camera] = []
                        depths[camera] = []
                    else:
                        assert (
                            intrinsics[camera] == obs["intrinsic_cv"]
                        ).all()

                    extrinsics[camera].append(obs["extrinsic_cv"])
                    rgbs[camera].append(rgb)
                    depths[camera].append(depth)

                joint_positions.append(frame_data["joint_action"])
                cartesian_positions.append(frame_data["endpose"])
                num_steps += 1

            for camera in camera_names:
                assert len(extrinsics[camera]) == num_steps
                extrinsics[camera] = np.stack(extrinsics[camera])

                assert len(rgbs[camera]) == num_steps
                assert len(depths[camera]) == num_steps
                for i, (rgb, depth) in enumerate(
                    zip(rgbs[camera], depths[camera], strict=False)
                ):
                    self.image_pack_file.write(f"{uuid}/{camera}/{i}", rgb)
                    self.depth_pack_file.write(f"{uuid}/{camera}/{i}", depth)

            assert len(joint_positions) == num_steps
            assert len(cartesian_positions) == num_steps
            joint_positions = np.stack(joint_positions)
            cartesian_positions = np.stack(cartesian_positions)

            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsics)
            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsics)
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions",
                joint_positions,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/cartesian_position",
                cartesian_positions,
            )
            self.meta_pack_file.write(f"{uuid}/camera_names", camera_names)

            index_data = dict(
                uuid=uuid,
                task_name=task_name,
                num_steps=num_steps,
                seeed=seed,
                camera_type=camera_type,
                simulation=True,
            )
            self.meta_pack_file.write(f"{uuid}/meta_data", index_data)
            self.write_index(ep_id, index_data)
            num_valid_ep += 1
            logger.info(
                f"finish process [{ep_id + 1}/{len(self.episodes)}] {uuid}, "
                f"num_steps:{num_steps} \n"
            )
        self.index_pack_file.write("__len__", num_valid_ep, commit=True)
        self.close()


if __name__ == "__main__":
    import argparse

    from robo_orchard_lab.utils import log_basic_config

    log_basic_config(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--seed_dir", type=str)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--embodiment", type=str, default=None)
    args = parser.parse_args()

    if args.task_names is None:
        task_names = None
    else:
        task_names = args.task_names.split(",")

    packer = RobotwinDataPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        seed_dir=args.seed_dir,
        task_names=task_names,
        embodiment=args.embodiment,
    )
    packer()
