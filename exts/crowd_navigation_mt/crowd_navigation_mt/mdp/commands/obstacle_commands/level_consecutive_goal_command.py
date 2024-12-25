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

from .goal_command_base import GoalCommandBaseTerm

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .level_consecutive_goal_command_cfg import LvlConsecutiveGoalCommandCfg


class LvlConsecutiveGoalCommand(GoalCommandBaseTerm):
    r"""Command that generates goal position commands based on terrain and defines the corresponding spawn locations.
    The goal commands are either sampled from RRT or from predefined fixed coordinates defined in the config.

    The goal coordinates/ commands are passed to the planners that generate the actual velocity commands.
    Goal coordinates are sampled in the world frame and then always transformed in the local robot frame.
    """

    cfg: LvlConsecutiveGoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: LvlConsecutiveGoalCommand, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- goal commands in base frame: (x, y, z)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)

        self.path_length_command = torch.zeros(self.num_envs, device=self.device)

        # -- run terrain analysis
        self._analysis = TerrainAnalysis(cfg=self.cfg.terrain_analysis, scene=self._env.scene)
        self._analysis.analyse()

        # -- fit kd-tree based on terrain level on all the graph nodes to quickly find the closest node to the obstacle
        pruning_tensor = torch.ones(self._analysis.points.shape[0], dtype=bool, device=self.device)
        pruning_tensor[self._analysis.isolated_points_ids] = False

        # Get grid dimensions
        self.num_levels, self.num_types, _ = env.scene.terrain.terrain_origins.shape

        # Expand terrain origins to [num_levels * num_types, 3] for computation
        terrain_origins_flat = env.scene.terrain.terrain_origins.view(-1, 3)  # Shape: [num_levels * num_types, 3]

        # Compute distances between points and terrain origins
        points_expanded = self._analysis.points.unsqueeze(1)  # Shape: [num_points, 1, 3]
        origins_expanded = terrain_origins_flat.unsqueeze(0)  # Shape: [1, num_levels * num_types, 3]
        distances = torch.sum((points_expanded - origins_expanded) ** 2, dim=-1)  # Shape: [num_points, num_levels * num_types]

        # Find the closest origin for each point
        closest_origin_indices = torch.argmin(distances, dim=1)  # Shape: [num_points]

        # Map flat indices back to grid (level, type)
        closest_levels = closest_origin_indices // self.num_types  # Shape: [num_points], terrain level indices (rows)
        closest_types = closest_origin_indices % self.num_types   # Shape: [num_points], terrain type indices (columns)

        # Create a tensor to hold grouped points
        self.grouped_points = []

        # Use masks to group points
        for level in range(self.num_levels):
            self.grouped_points.append([])
            for _type in range(self.num_types):
                # Mask for points belonging to the current level and type
                terrain_mask = (closest_levels == level) & (closest_types == _type)
                points_in_cell = self._analysis.points[terrain_mask]  # Extract points for the cell
                # Concatenate the grouped points
                self.grouped_points[level].append(points_in_cell)
        
        
        self._grouped_kd_trees = []
        for level in range(self.num_levels):
            self._grouped_kd_trees.append([])
            for type in range(self.num_types):
                self._grouped_kd_trees[level].append(KDTree(self.grouped_points[level][type].cpu().numpy()))

        if self.cfg.plot_points:
            self.plot_grouped_points()        

        # # -- fit kd-tree on all the graph nodes to quickly find the closest node to the robot
        # pruning_tensor = torch.ones(self._analysis.points.shape[0], dtype=bool, device=self.device)
        # pruning_tensor[self._analysis.isolated_points_ids] = False
        # self._kd_tree = KDTree(self._analysis.points[pruning_tensor].cpu().numpy())
        # self._mapping_kd_tree_to_graph = torch.arange(pruning_tensor.sum(), device=self.device)
        # # the cumulative sum skips over the isolated points
        # self._mapping_kd_tree_to_graph += torch.cumsum(~pruning_tensor, 0)[pruning_tensor]

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "LvlConsecutiveGoalCommandGenerator:\n"
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

    @property
    def analysis(self) -> TerrainAnalysis:
        """The terrain analysis object."""
        return self._analysis

    """
    Implementation specific functions.
    """
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal commands for the specified environments.

        Args:
            env_ids (Sequence[int]): The list of environment IDs to resample.
        """
         # get the robot position for the environment
        robot_pos = self.robot.data.root_pos_w[env_ids, :3]

        # only the commands from the active dynamic obstacles are resampled
        # the other stay beneath the ground and dont get any new commands/ actions
        env_ids = env_ids[env_ids <= (self.num_levels * ((self.num_types * (self.num_types + 1) / 2) * 5) - 1)]

        # sample a goal from the samples generated from the graph
        for env_id in env_ids:
            # TODO: make level and type as an attribute of the obstacle
            level = int(env_id // ((self.num_types * (self.num_types + 1) / 2) * 5))
            id_per_level = (env_id % ((self.num_types * (self.num_types + 1) / 2) * 5)) 
            _type = int((torch.sqrt(1 + 8 * (id_per_level / 5)) - 1) // 2)
            
            # TODO @ vairaviv this should never be the case
            if level > self.num_levels or _type > self.num_types:
                print("level and type are out of bound, check in level_consecutive_goal_command.py line 157")
                continue

            random_point_idx = torch.randperm(self.grouped_points[level][_type].size(0))[0]
            self.pos_command_w[env_id] = self.grouped_points[level][_type][random_point_idx]

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
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)

    """
    Helper functions
    """

    def plot_grouped_points(self):
        # Create a figure
        plt.figure(figsize=(10, 8))

        # Generate a unique color for each level/type combination
        colors = cm.get_cmap("tab20", self.num_levels * self.num_types)

        for level in range(self.num_levels):
            for _type in range(self.num_types):
                points = self.grouped_points[level][_type].cpu()
                x, y = points[:, 0], points[:, 1]  # Extract x and y coordinates
                label = f"Level {level}, Type {_type}"
                color_idx = level * self.num_types + _type
                plt.scatter(x, y, color=colors(color_idx), label=label, alpha=0.7)

        # Add legend, title, and labels
        plt.legend(loc="best")
        plt.title("Points Grouped by Terrain Type and Level")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Save the plot as an image
        plt.savefig("logs/plots/terrain_points_plot.png")
        print("Plot saved as 'logs/plots/terrain_points_plot.png'.")