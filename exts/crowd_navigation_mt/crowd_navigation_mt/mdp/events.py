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

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    # from omni.isaac.lab_tasks.navigation.cost_cloud_nav.mdp.commands.goal_command import GoalCommand
    from crowd_navigation_mt.mdp.commands import RobotGoalCommand


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
