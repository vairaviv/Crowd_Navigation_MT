# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the common functions that can be used to enable different randomizations.

Randomization includes anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.orbit.managers.RandomizationTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import time

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from omni.isaac.lab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from omni.isaac.lab.terrains import TerrainImporter

    # from omni.isaac.lab_tasks.navigation.cost_cloud_nav.mdp.commands.goal_command import GoalCommand
    from crowd_navigation_mt.mdp.commands import RobotGoalCommand, LvlConsecutiveGoalCommand, SemanticGoalCommand


def reset_root_state_from_terrain_lvl_type(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "sfm_target_pos",
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command: LvlConsecutiveGoalCommand = env.command_manager._terms[command_name]

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_position_semantic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    additive_heading_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "robot_goal",
):
    """Reset the asset root state to the spawn state defined by the command generator.

    Args:
        env: The environment object.
        env_ids: The environment ids to reset.
        additive_heading_range: The additive heading range to apply to the spawn heading.
        asset_cfg: The asset configuration to reset. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: SemanticGoalCommand = env.command_manager._terms[command_name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions - based on given start points (command generator)
    # pos_offset = torch.zeros_like(root_states[:, 0:3])

    # get valid spawn locations from the command
    random_idx = torch.randint(0, goal_cmd_geneator.valid_pos_idx.size(0), (len(env_ids),), device=goal_cmd_geneator.device)
    positions = goal_cmd_geneator.valid_pos_w[random_idx, :]

    # overwrite spawn position in goal command for other calculations
    goal_cmd_geneator.pos_spawn_w[env_ids] = positions

    # TODO: @vairaviv remove after debugging
    if torch.any(
        torch.isclose(positions[:, 0], goal_cmd_geneator.pos_command_w[env_ids, 0], 1e-4) & 
        torch.isclose(positions[:, 1], goal_cmd_geneator.pos_command_w[env_ids, 1], 1e-4)
    ):
        print("[DEBUG]: Position Command and Spawn location are the same!")
        time.sleep(10)
        

    # positions = goal_cmd_geneator.pos_spawn_w[env_ids]
    # positions = env.scene.env_origins[env_ids] + pos_offset
    # orientations - based on given start points (command generator)
    euler_angles = torch.zeros_like(root_states[:, 3:6])
    euler_angles[:, 2].uniform_(*additive_heading_range.get("yaw", (0.0, 0.0)))  # add additive disturbance to heading
    euler_angles[:, 2] += goal_cmd_geneator.heading_spawn_w[env_ids]
    orientations = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    # velocities - zero
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_position_plr(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    additive_heading_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "robot_goal",
):
    """Reset the asset root state to the spawn state defined by the command generator.

    Args:
        env: The environment object.
        env_ids: The environment ids to reset.
        additive_heading_range: The additive heading range to apply to the spawn heading.
        asset_cfg: The asset configuration to reset. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions - based on given start points (command generator)
    # pos_offset = torch.zeros_like(root_states[:, 0:3])
    positions = goal_cmd_geneator.pos_spawn_w[env_ids]
    # positions = env.scene.env_origins[env_ids] + pos_offset
    # orientations - based on given start points (command generator)
    euler_angles = torch.zeros_like(root_states[:, 3:6])
    euler_angles[:, 2].uniform_(*additive_heading_range.get("yaw", (0.0, 0.0)))  # add additive disturbance to heading
    euler_angles[:, 2] += goal_cmd_geneator.heading_spawn_w[env_ids]
    orientations = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    # velocities - zero
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_position_regular_all_terrains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    additive_heading_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "robot_goal",
):
    """Reset the asset root state to the spawn state defined by the command generator.

    Args:
        env: The environment object.
        env_ids: The environment ids to reset.
        additive_heading_range: The additive heading range to apply to the spawn heading.
        asset_cfg: The asset configuration to reset. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions - based on given start points (command generator)
    pos_offset = torch.zeros_like(root_states[:, 0:3])
    pos_offset += goal_cmd_geneator.pos_spawn_w[env_ids]
    positions = env.scene.terrain.terrain_origins.view(-1, 3)[env_ids] + pos_offset
    # orientations - based on given start points (command generator)
    euler_angles = torch.zeros_like(root_states[:, 3:6])
    euler_angles[:, 2].uniform_(*additive_heading_range.get("yaw", (0.0, 0.0)))  # add additive disturbance to heading
    euler_angles[:, 2] += goal_cmd_geneator.heading_spawn_w[env_ids]
    orientations = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    # velocities - zero
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
