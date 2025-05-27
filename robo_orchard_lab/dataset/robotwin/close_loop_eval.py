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

import copy
import logging
import os

import cv2
import numpy as np
import torch

logger = logging.getLogger(__file__)


def close_loop_data_process(dataset, obs, task_name, joint_state):
    images = []
    depths = []
    T_world2cam = []
    intrinsic = []
    for camera_data in obs["observation"].values():
        if dataset.load_image:
            images.append(camera_data["rgb"])
        if dataset.load_depth:
            depths.append(camera_data["depth"] / 1000)

        _tmp = np.eye(4)
        _tmp[:3] = camera_data["extrinsic_cv"]
        T_world2cam.append(_tmp)

        _tmp = np.eye(4)
        _tmp[:3, :3] = camera_data["intrinsic_cv"]
        intrinsic.append(_tmp)

    joint_state.append(obs["joint_action"])
    data = dict(
        intrinsic=np.stack(intrinsic),
        T_world2cam=np.stack(T_world2cam),
        T_base2world=copy.deepcopy(dataset.T_base2world),
        step_index=len(joint_state) - 1,
        joint_state=np.stack(joint_state),
    )
    if dataset.T_base2ego is not None:
        data["T_base2ego"] = copy.deepcopy(dataset.T_base2ego)
    if dataset.load_image:
        data["imgs"] = np.stack(images)
    if dataset.load_depth:
        data["depths"] = np.stack(depths)

    instructions = dataset.instructions[task_name]
    if isinstance(instructions, str):
        text = instructions
    elif len(instructions) == 0:
        text = ""
    else:
        text = instructions[0]
    data["text"] = text

    for transform in dataset.transforms:
        if transform is None:
            continue
        data = transform(data)
    for k, v in data.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            data[k] = v[None]
        else:
            data[k] = [v]
    return data, joint_state


def evaluation(env, model, dataset, seed, save_path=None):
    task_name = env.__class__.__name__
    num_left_joints = len(env.left_arm_joint_id)
    num_right_joints = len(env.right_arm_joint_id)

    env._update_render()
    if env.render_freq:
        env.viewer.render()
    env.actor_pose = True
    success_flag = False
    left_gripper_joint_id = [34, 35]
    right_gripper_joint_id = [36, 37]
    gripper_velocity = 0.05

    video_writer = None

    def visualize(data, video_writer):
        if save_path is None:
            return
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        imgs = data["imgs"].cpu().numpy()[0]
        imgs = np.concatenate(
            [
                np.concatenate([imgs[0], imgs[3]], axis=1),
                np.concatenate([imgs[1], imgs[2]], axis=1),
            ],
            axis=0,
        )

        env.observer_camera.take_picture()
        observer_img = env._get_camera_rgba(env.observer_camera)[..., :3]
        scale = imgs.shape[0] / observer_img.shape[0]
        observer_img = cv2.resize(
            observer_img, (int(scale * observer_img.shape[1]), imgs.shape[0])
        )
        imgs = np.concatenate([imgs, observer_img], axis=1)

        if video_writer is None:
            file = os.path.join(save_path, f"{task_name}_seed{seed}.mp4")
            logger.info(f"visualization result: {file}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                file,
                fourcc,
                20,
                imgs.shape[:2][::-1],
            )
        video_writer.write(np.uint8(imgs)[..., ::-1])
        return video_writer

    step_cnt = 0
    joint_state = []
    while step_cnt < env.step_lim:
        obs = env.get_obs()
        data, joint_state = close_loop_data_process(
            dataset, obs, task_name, joint_state
        )
        video_writer = visualize(data, video_writer)
        actions = model(data)[0]["pred_actions"][0]

        valid_action_step = 32
        actions = actions[:valid_action_step, :, 0].cpu().numpy()
        current_joint_state = obs["joint_action"]
        actions = np.concatenate([current_joint_state[None], actions])

        left_gripper_pos = actions[:, num_left_joints]
        right_gripper_pos = actions[:, num_left_joints + num_right_joints + 1]
        left_arm_pos = actions[:, :num_left_joints]
        right_arm_pos = actions[
            :, num_left_joints + 1 : num_left_joints + num_right_joints + 1
        ]

        try:
            left_arm_pos, left_arm_vel = env.left_planner.TOPP(
                left_arm_pos, 1 / 250, verbose=True
            )[1:3]
            left_n_step = left_arm_pos.shape[0]
            left_step_idx = 0
        except Exception:
            left_n_step = left_step_idx = 1

        try:
            right_arm_pos, right_arm_vel = env.right_planner.TOPP(
                right_arm_pos,
                1 / 250,
                verbose=True,
            )[1:3]
            right_n_step = right_arm_pos.shape[0]
            right_step_idx = 0
        except Exception:
            right_n_step = right_step_idx = 1

        n_step = max(left_n_step, right_n_step)
        action_freq = n_step // actions.shape[0]

        step_idx = 0
        while left_step_idx < left_n_step or right_step_idx < right_n_step:
            qf = env.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            env.robot.set_qf(qf)
            if left_step_idx / left_n_step <= right_step_idx / right_n_step:
                for j in range(num_left_joints):
                    left_j = env.left_arm_joint_id[j]
                    env.active_joints[left_j].set_drive_target(
                        left_arm_pos[left_step_idx][j]
                    )
                    env.active_joints[left_j].set_drive_velocity_target(
                        left_arm_vel[left_step_idx][j]
                    )
                if not env.fix_gripper:
                    for joint_id in left_gripper_joint_id:
                        gripper_idx = min(
                            left_step_idx // action_freq,
                            left_gripper_pos.shape[0] - 1,
                        )
                        env.active_joints[joint_id].set_drive_target(
                            left_gripper_pos[gripper_idx]
                        )
                        env.active_joints[joint_id].set_drive_velocity_target(
                            gripper_velocity
                        )
                        env.left_gripper_val = left_gripper_pos[gripper_idx]
                left_step_idx += 1

            if left_step_idx / left_n_step >= right_step_idx / right_n_step:
                for j in range(num_right_joints):
                    right_j = env.right_arm_joint_id[j]
                    env.active_joints[right_j].set_drive_target(
                        right_arm_pos[right_step_idx][j]
                    )
                    env.active_joints[right_j].set_drive_velocity_target(
                        right_arm_vel[right_step_idx][j]
                    )
                if not env.fix_gripper:
                    for joint_id in right_gripper_joint_id:
                        gripper_idx = min(
                            right_step_idx // action_freq,
                            right_gripper_pos.shape[0] - 1,
                        )
                        env.active_joints[joint_id].set_drive_target(
                            right_gripper_pos[gripper_idx]
                        )
                        env.active_joints[joint_id].set_drive_velocity_target(
                            gripper_velocity
                        )
                        env.right_gripper_val = right_gripper_pos[gripper_idx]
                right_step_idx += 1

            env.scene.step()
            env._update_render()
            if (step_idx + 1) % action_freq == 0:
                if env.render_freq:
                    env.viewer.render()
                obs = env.get_obs()
                data, joint_state = close_loop_data_process(
                    dataset, obs, task_name, joint_state
                )
                video_writer = visualize(data, video_writer)
                step_cnt += 1

            step_idx += 1
            if env.check_success():
                success_flag = True
                break

            if not env.actor_pose:
                break

            if step_idx >= action_freq * valid_action_step:
                break

        if step_idx % max(action_freq, 1) != 0:
            env._update_render()
            if env.render_freq:
                env.viewer.render()
            obs = env.get_obs()
            data, joint_state = close_loop_data_process(
                dataset, obs, task_name, joint_state
            )
            video_writer = visualize(data, video_writer)
            step_cnt += 1

        logger.info(f"step: {step_cnt} / {env.step_lim}")
        if success_flag:
            if video_writer is not None:
                video_writer.release()
            logger.info(f"seed {seed}: success!")
            return True

        if not env.actor_pose:
            break

    if video_writer is not None:
        video_writer.release()
    logger.info(f"seed {seed}: fail!")
    return False
