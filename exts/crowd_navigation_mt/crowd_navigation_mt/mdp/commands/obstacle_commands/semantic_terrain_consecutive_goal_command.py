# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from scipy.spatial import KDTree
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

from nav_collectors.terrain_analysis import TerrainAnalysis

from .semantic_terrain_goal_command import SemanticGoalCommand

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .semantic_terrain_consecutive_goal_command_cfg import SemanticConsecutiveGoalCommandCfg


class SemanticConsecutiveGoalCommand(SemanticGoalCommand):
    r"""Command that generates goal position commands based on the semantic terrain.
    """

    cfg: SemanticConsecutiveGoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: SemanticConsecutiveGoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        super().__init__(cfg, env)    
        

        # -- metrics
        self.metrics.clear()
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "SemanticConsecutiveGoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b

    """
    Implementation specific functions.
    """
    def _resample_command(self, env_ids: Sequence[int]):
        """sample new goal positions.
        It is sampled on a sidewalk and randomly selected from the valid positions."""

        if self.grid_map is None:
            self.pos_command_w[env_ids, :2] = self._env.scene.env_origins[env_ids, :2]
        # else:
        #     random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
        #     self.pos_command_w[env_ids, :2] = self.valid_pos_w[random_idx, :2]
        else:  
            idx = self.point_kd_tree.query_ball_point(self.robot.data.root_pos_w[env_ids, :2].cpu(), r=self.radius_lvl[env_ids].cpu())
            self.pos_spawn_w[env_ids, :2] = self.robot.data.root_pos_w[env_ids, :2]
            #random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
            try:
                random_idx = torch.tensor(
                    [
                        list[torch.randint(0, len(list), (1,)).item()] 
                        if len(list) > 0 
                        else (
                            print(f"[INFO]: SemanticConsecutiveGoalCommand: Empty list encountered, using self.valid_pos_w instead") or
                            torch.randint(0, self.valid_pos_w.shape[0], (1,)).item()
                        ) # evalutates both and easier for debugging
                        for list in idx
                    ]
                ).to(device=self.device)
                # if self.pos_spawn_w[]
            except IndexError:
                print("[DEBUG]: SemanticConsecutiveGoalCommand: the root position of the robot is at an invalid position")
                random_idx = torch.randint(0, self.valid_pos_w.shape[0], (len(env_ids),))
            
            if torch.any(
                torch.isclose(self.valid_pos_w[random_idx, 0], self.pos_spawn_w[env_ids, 0], 1e-4) & 
                torch.isclose(self.valid_pos_w[random_idx, 1], self.pos_spawn_w[env_ids, 1], 1e-4)
            ):
                mask = (
                    torch.isclose(self.valid_pos_w[random_idx, 0], self.pos_spawn_w[env_ids, 0], 1e-4) & 
                    torch.isclose(self.valid_pos_w[random_idx, 1], self.pos_spawn_w[env_ids, 1], 1e-4)
                ).to(device=self.device)
                random_idx[mask] -= 1  # this will never be out of index as list[-1] is valid too
                print("[DEBUG]: SemanticConsecutiveGoalCommand: Position Command and Spawn location are the same!")
            self.pos_command_w[env_ids, :2] = self.valid_pos_w[random_idx, :2] 
            
    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component

        # update commands which are close to the goal
        goal_dist = torch.norm(target_vec, dim=1)
        close_goal = goal_dist < self.cfg.resample_distance_threshold
        if torch.any(close_goal):
            self._resample_command(torch.where(close_goal)[0])
            target_vec[close_goal] = self.pos_command_w[close_goal] - self.robot.data.root_pos_w[close_goal, :3]
            target_vec[close_goal, 2] = 0.0  # ignore z component

        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

    def _update_metrics(self):
        """Update metrics."""
        # TODO: only the obstacles actuated are regarded here, parametrize!
        num_tot_agent = self.cfg.num_sfm_obstacle
        self.metrics["error_pos"][:num_tot_agent] = torch.norm(
            self.pos_command_w[:num_tot_agent, :] - self.robot.data.root_pos_w[:num_tot_agent, :3], dim=1
        )