#!/usr/bin/env python3

"""
Complete script to convert LMDB data to HDF5 format for ACT training.
Combines functionality from convert2ACT.py and example usage.
"""

import argparse
import copy
import h5py
import json
import logging
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class RoboTwinLmdbDataset(BaseLmdbManipulationDataset):
    """RoboTwin LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├── uuid: str
            ├── task_name: str
            ├── num_steps: int
            └── simulation: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data: dict
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └── {cam_name}: np.ndarray[num_steps x 4 x 4]
        {uuid}/intrinsic
            ├── {cam_name}: np.ndarray[3 x 3]
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/joint_positions

    Image storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}

    Depth storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}
    """

    DEFAULT_INSTRUCTIONS = {
        "block_hammer_beat": "Hit the block with a hammer.",
        "block_handover": "Left arm picks up a block and passes it to the right arm, which places it on the blue mat.",  # noqa: E501
        "blocks_stack_easy": "Stack red cubes first, then black cubes in a specific spot.",  # noqa: E501
        "blocks_stack_hard": "Stack red cubes first, then green, then blue cubes in a specific spot.",  # noqa: E501
        "bottle_adjust": "Pick up the bottle so its head faces up.",
        "container_place": "Move containers to a fixed plate.",
        "diverse_bottles_pick": "Lift two bottles with different designs to a designated location using both arms.",  # noqa: E501
        "dual_bottles_pick_easy": "Pick up a red bottle from the left and a green bottle from the right, and move them together to a designated location.",  # noqa: E501
        "dual_bottles_pick_hard": "Pick up a red bottle from the left and a green bottle from the right in any position, and move them together to a designated location.",  # noqa: E501
        "dual_shoes_place": "Pick up shoes from each side and place them in a blue area with heads facing left.",  # noqa: E501
        "empty_cup_place": "Place an empty cup on a cup mat.",
        "mug_hanging_easy": "Move a mug to the middle and hang it on a fixed rack.",  # noqa: E501
        "mug_hanging_hard": "Move a mug to the middle and hang it on a randomly placed rack.",  # noqa: E501
        "pick_apple_messy": "Pick up an apple from among other items.",
        "put_apple_cabinet": "Open a cabinet and place an apple inside.",
        "shoe_place": "Move shoes to the shoebox or the blue area with heads facing left.",  # noqa: E501
        "tool_adjust": "Pick up a tool so its head faces up.",
        "put_bottles_dustbin": "Put the bottles into the dustbin.",
        "bowls_stack": "Stack the bowls together and put them in a specific spot.",  # noqa: E501
        "classify_tactile": "Classify the block according to its shapes and put them in the corresponding positions.",  # noqa: E501
        "others": "Complete all the tasks you see",
    }

    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=True,
        task_names=None,
        lazy_init=False,
        num_episode=None,
        cam_names=None,
        T_base2world=None,  # noqa: N803
        T_base2ego=None,  # noqa: N803
        default_space="base",
        instructions=None,
        instruction_keys=("seen", "unseen"),
    ):
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            num_episode=num_episode,
        )
        self.cam_names = cam_names
        if T_base2world is None:
            logger.warning("dataset T_base2world is not set, use default.")
            T_base2world = np.array(  # noqa: N806
                [
                    [0, -1, 0, 0],
                    [1, 0, 0, -0.65],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        elif isinstance(T_base2world, List):
            T_base2world = np.array(T_base2world)  # noqa: N806
        self.T_base2world = T_base2world
        self.T_base2ego = T_base2ego
        assert default_space in ["base", "world", "ego"]
        self.default_space = default_space
        self.load_instructions(instructions)
        self.instruction_keys = instruction_keys

    def load_instructions(self, instructions):
        if instructions is None:
            self.instructions = self.DEFAULT_INSTRUCTIONS
        elif os.path.isfile(instructions):
            assert instructions.endswith(".json")
            self.instructions = json.load(open(instructions, "r"))
        else:
            assert isinstance(instructions, dict)
            self.instructions = instructions

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        task_name = idx_data.task_name
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        _T_world2cam = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]  # noqa: N806
        _intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        T_world2cam = []  # noqa: N806
        intrinsic = []
        for cam_name in cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(image, bytes):
                    image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                images.append(image)
            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(depth, bytes):
                    depth = np.frombuffer(depth, np.uint8)
                depth = (
                    cv2.imdecode(
                        depth,
                        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
                    )
                    / 1000
                )
                depths.append(depth)

            _tmp = np.eye(4)
            if _T_world2cam[cam_name].ndim == 3:  # for dynamic camera
                _tmp[:3] = _T_world2cam[cam_name][step_index][:3]
            else:  # ndim == 2, for fixed camera
                _tmp[:3] = _T_world2cam[cam_name][:3]
            T_world2cam.append(_tmp)

            _tmp = np.eye(4)
            _tmp[:3, :3] = _intrinsic[cam_name][:3, :3]
            intrinsic.append(_tmp)

        if self.load_image:
            images = np.stack(images)
        if self.load_depth:
            depths = np.stack(depths)
        T_world2cam = np.stack(T_world2cam)  # noqa: N806
        intrinsic = np.stack(intrinsic)

        joint_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        ee_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/cartesian_position"
        ]
        if ee_state.ndim == 3:
            ee_state = ee_state.reshape(ee_state.shape[0], -1)

        data = dict(
            uuid=uuid,
            step_index=step_index,
            intrinsic=intrinsic,
            T_world2cam=T_world2cam,
            T_base2world=copy.deepcopy(self.T_base2world),
            joint_state=joint_state,
            ee_state=ee_state,
        )
        if self.T_base2ego is not None:
            data["T_base2ego"] = copy.deepcopy(self.T_base2ego)
        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        instructions = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if instructions is None:
            meta_data = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
            instructions = meta_data.get("instruction")

        if instructions is None:
            instructions = self.instructions.get(
                task_name,
                self.DEFAULT_INSTRUCTIONS["others"],
            )
        elif isinstance(instructions, dict):
            _tmp = []
            for k in self.instruction_keys:
                if isinstance(instructions[k], str):
                    _tmp.append(instructions[k])
                else:
                    _tmp.extend(instructions[k])
            instructions = _tmp

        if isinstance(instructions, str):
            text = instructions
        elif len(instructions) == 0:
            text = ""
        else:
            idx = np.random.randint(len(instructions))
            text = instructions[idx]
        data["text"] = text
        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data


def convert_lmdb_to_hdf5(lmdb_paths, output_path, start_index=0, camera_names=None):
    """
    Convert LMDB format data to HDF5 format for ACT training.
    
    Args:
        lmdb_paths: List of paths to LMDB datasets
        output_path: Output directory for HDF5 files
        start_index: Starting episode index
        camera_names: List of camera names to use (default: auto-detect)
        
    Returns:
        max_episode_len: Maximum episode length across all episodes
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    index = start_index
    max_episode_len = 0
    
    # Initialize dataset
    dataset = RoboTwinLmdbDataset(
        paths=lmdb_paths,
        load_image=True,
        load_depth=False,
        cam_names=camera_names,
        lazy_init=False
    )
    
    logger.info(f"Total episodes in dataset: {len(dataset.episode_indices)}")
    
    for episode_idx in tqdm(range(len(dataset.episode_indices)), desc="Converting episodes"):
        try:
            # Get episode info
            lmdb_index = dataset.lmdb_indices[episode_idx]
            episode_index = dataset.episode_indices[episode_idx]
            num_steps = dataset.num_steps[episode_idx]
            
            # Get episode data
            idx_data = BaseIndexData.model_validate(
                dataset.idx_lmdbs[lmdb_index][episode_index]
            )
            uuid = idx_data.uuid
            task_name = idx_data.task_name
            
            # Get camera names
            if camera_names is not None:
                cam_names = camera_names
            else:
                cam_names = dataset.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]
            
            logger.info(f"Processing episode {episode_idx}: {uuid}, task: {task_name}, steps: {num_steps}")
            logger.info(f"Camera names: {cam_names}")
            
            # Collect all data for this episode
            episode_images = {cam_name: [] for cam_name in cam_names}
            joint_states = []
            ee_states = []
            
            # Get start and end indices for this episode
            if episode_idx == 0:
                start_step = 0
            else:
                start_step = dataset.cumsum_steps[episode_idx - 1]
            end_step = dataset.cumsum_steps[episode_idx]
            
            # Collect data for each step
            for step_idx in range(start_step, end_step):
                data = dataset[step_idx]
                
                # Collect images
                if 'imgs' in data:
                    for i, cam_name in enumerate(cam_names):
                        episode_images[cam_name].append(data['imgs'][i])
                
                # Collect joint states
                if 'joint_state' in data:
                    joint_states.append(data['joint_state'][step_idx - start_step])
                
                if 'ee_state' in data:
                    ee_states.append(data['ee_state'][step_idx - start_step])
            
            # Create HDF5 file
            hdf5_output_path = os.path.join(output_path, f"episode_{index}.hdf5")
            
            with h5py.File(hdf5_output_path, "w") as f:
                # Convert joint states to qpos format
                joint_states = np.array(joint_states)
                
                # For dual-arm setup, assume 14-dim: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)
                # If joint_states doesn't match expected dimensions, pad or truncate
                if joint_states.shape[1] < 14:
                    # Pad with zeros
                    padding = np.zeros((joint_states.shape[0], 14 - joint_states.shape[1]))
                    qpos = np.concatenate([joint_states, padding], axis=1)
                elif joint_states.shape[1] > 14:
                    # Truncate to 14 dimensions
                    qpos = joint_states[:, :14]
                else:
                    qpos = joint_states
                
                # Generate actions (next step qpos)
                actions = []
                for i in range(len(qpos) - 1):
                    actions.append(qpos[i + 1])
                
                # Last action is zero (end of episode)
                last_action = np.zeros(14, dtype=np.float32)
                actions.append(last_action)
                actions = np.array(actions, dtype=np.float32)
                
                # Save action data
                f.create_dataset('action', data=actions, dtype="float32")
                
                # Create observations group
                obs = f.create_group("observations")
                obs.create_dataset('qpos', data=qpos.astype(np.float32), dtype="float32")
                obs.create_dataset("left_arm_dim", data=np.array(6))
                obs.create_dataset("right_arm_dim", data=np.array(6))
                
                # Create images group
                images = obs.create_group("images")
                
                # Save images based on camera configuration
                if len(cam_names) == 3:
                    # Dual-arm format with 3 cameras
                    cam_mapping = {
                        "cam_high": cam_names[0] if "head" in cam_names[0] or "high" in cam_names[0] else cam_names[0],
                        "cam_left_wrist": None,
                        "cam_right_wrist": None
                    }
                    
                    # Try to identify wrist cameras
                    for cam_name in cam_names[1:]:
                        if "left" in cam_name.lower() and "wrist" in cam_name.lower():
                            cam_mapping["cam_left_wrist"] = cam_name
                        elif "right" in cam_name.lower() and "wrist" in cam_name.lower():
                            cam_mapping["cam_right_wrist"] = cam_name
                        elif cam_mapping["cam_left_wrist"] is None:
                            cam_mapping["cam_left_wrist"] = cam_name
                        elif cam_mapping["cam_right_wrist"] is None:
                            cam_mapping["cam_right_wrist"] = cam_name
                    
                    # Save images for each camera
                    for hdf5_cam_name, lmdb_cam_name in cam_mapping.items():
                        if lmdb_cam_name is not None and lmdb_cam_name in episode_images:
                            cam_images = np.stack(episode_images[lmdb_cam_name])
                            images.create_dataset(hdf5_cam_name, data=cam_images, dtype=np.uint8)
                            logger.info(f"Saved {hdf5_cam_name} images: {cam_images.shape}")
                else:
                    # Save all cameras with their original names
                    for cam_name in cam_names:
                        if cam_name in episode_images:
                            cam_images = np.stack(episode_images[cam_name])
                            images.create_dataset(cam_name, data=cam_images, dtype=np.uint8)
                            logger.info(f"Saved {cam_name} images: {cam_images.shape}")
            
            max_episode_len = max(max_episode_len, num_steps)
            logger.info(f"Converted episode {episode_idx} to {hdf5_output_path}")
            index += 1
            
        except Exception as e:
            logger.error(f"Failed to convert episode {episode_idx}: {e}")
            continue
    
    return max_episode_len


def run_example():
    """Run example conversion with predefined settings."""
    # Example usage
    lmdb_paths = ["/home/kwj/GitCode/RoboOrchardLab/Real-World-Dataset/lmdb_dataset_stack_bowls_three_2025_09_09"]
    output_path = "/home/kwj/GitCode/RoboOrchardLab/converted_data/stack_bowls_three"
    task_name = "stack_bowls_three"
    
    # Convert data
    print(f"Converting LMDB data from: {lmdb_paths}")
    print(f"Output directory: {output_path}")
    print(f"Task name: {task_name}")
    
    try:
        max_episode_len = convert_lmdb_to_hdf5(
            lmdb_paths=lmdb_paths,
            output_path=output_path,
            start_index=0,
            camera_names=None  # Auto-detect camera names
        )
        
        # Count episodes
        if os.path.exists(output_path):
            expert_data_num = len([f for f in os.listdir(output_path) if f.endswith('.hdf5')])
        else:
            expert_data_num = 0
        
        print(f"\n=== Conversion Summary ===")
        print(f"Task: {task_name}")
        print(f"Number of episodes: {expert_data_num}")
        print(f"Max episode length: {max_episode_len}")
        print(f"Output directory: {output_path}")
        
        # Create config file
        config_path = os.path.join(output_path, "SIM_TASK_CONFIGS.json")
        config = {
            f"sim-{task_name}": {
                "dataset_dir": output_path,
                "num_episodes": int(expert_data_num),
                "episode_len": int(max_episode_len),
                "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],  # dual-arm format
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Config saved to: {config_path}")
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to convert LMDB data to HDF5 format."""
    parser = argparse.ArgumentParser(
        description='Convert LMDB format data to HDF5 for ACT training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with example data
  python example_convert.py

  # Convert custom data
  python example_convert.py --lmdb_paths /path/to/lmdb --output_path /path/to/output --task_name my_task

  # Convert with specific cameras
  python example_convert.py --lmdb_paths /path/to/lmdb --output_path /path/to/output --task_name my_task --camera_names cam1 cam2 cam3
        """
    )
    parser.add_argument('--lmdb_paths', type=str, nargs='+', default=None,
                        help='Paths to LMDB datasets (if not provided, runs example)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output directory for HDF5 files')
    parser.add_argument('--task_name', type=str, default=None,
                        help='Task name for the dataset')
    parser.add_argument('--camera_names', type=str, nargs='*', default=None,
                        help='Camera names to use (auto-detect if not specified)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting episode index')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # If no arguments provided, run example
    if args.lmdb_paths is None or args.output_path is None or args.task_name is None:
        print("No arguments provided, running example conversion...")
        run_example()
        return
    
    # Convert data
    max_episode_len = convert_lmdb_to_hdf5(
        lmdb_paths=args.lmdb_paths,
        output_path=args.output_path,
        start_index=args.start_index,
        camera_names=args.camera_names
    )
    
    # Count number of episodes
    expert_data_num = len([f for f in os.listdir(args.output_path) if f.endswith('.hdf5')])
    
    # Create SIM_TASK_CONFIGS.json
    sim_task_configs_path = os.path.join(args.output_path, "SIM_TASK_CONFIGS.json")
    
    # Determine camera names for config
    if args.camera_names and len(args.camera_names) == 3:
        config_camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    elif args.camera_names:
        config_camera_names = args.camera_names
    else:
        config_camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]  # default dual-arm
    
    sim_task_configs = {
        f"sim-{args.task_name}": {
            "dataset_dir": args.output_path,
            "num_episodes": int(expert_data_num),
            "episode_len": int(max_episode_len),
            "camera_names": config_camera_names,
        }
    }
    
    with open(sim_task_configs_path, "w") as f:
        json.dump(sim_task_configs, f, indent=4)
    
    logger.info(f"Saved SIM_TASK_CONFIGS.json to {sim_task_configs_path}")
    logger.info(f"Task: {args.task_name}, Episodes: {expert_data_num}, Max Episode Length: {max_episode_len}")
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()