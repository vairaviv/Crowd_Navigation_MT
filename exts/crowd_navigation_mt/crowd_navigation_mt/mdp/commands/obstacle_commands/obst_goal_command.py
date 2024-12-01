# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate, quat_rotate_inverse, wrap_to_pi, yaw_quat

from nav_collectors.collectors import TrajectorySampling
from nav_collectors.terrain_analysis import TerrainAnalysis

from .goal_command_base import CYLINDER_MARKER_CFG, GoalCommandBaseTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .obst_goal_command_cfg import ObstGoalCommandCfg


class ObstGoalCommand(GoalCommandBaseTerm):
    r"""Command that generates goal position commands based on terrain and defines the corresponding spawn locations.
    The goal commands are either sampled from RRT or from predefined fixed coordinates defined in the config.

    The goal coordinates/ commands are passed to the planners that generate the actual velocity commands.
    Goal coordinates are sampled in the world frame and then always transformed in the local robot frame.
    """

    cfg: ObstGoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: ObstGoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # -- goal commands in base frame: (x, y, z)
        # self.pos_command_b = torch.zeros_like(self.pos_command_w)

        # -- goal commands in base frame: (x, y)
        self.pos_command_b = torch.zeros(self.num_envs, 2, device=self.device)

        # -- heading command
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # -- path length of the start-goal pairs
        self.path_length_command = torch.zeros(self.num_envs, device=self.device)

        # -- spawn locations (x, y, z, heading)
        self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)

        # -- define number of path to sample
        self.num_paths = cfg.trajectory_config["num_paths"]
        self.min_path_length = cfg.trajectory_config["min_path_length"]
        self.max_path_length = cfg.trajectory_config["max_path_length"]

        # -- run terrain analysis and sample first trajectories
        self.traj_sampling = TrajectorySampling(cfg=self.cfg.traj_sampling, scene=self._env.scene)
        self.sample_trajectories()

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

        # -- evaluation - monitor not updated environments
        self.not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nb_sampled_paths = 0

        # -- tracking when the trajectory config was last updated, for use in curriculum updates.
        self.last_update_config_env_step = 0

        # resampling is handled on CommandTerm level, dont know why it does not work
        # # -- buffer for resampling time 
        # self.resampling_time = resampling_times = torch.rand(
        #     self._env.num_envs, dtype=torch.float32, device=self.device)

    def __str__(self) -> str:
        msg = "GoalCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base pose in base frame. Shape is (num_envs, 7)."""
        # return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(-1)), dim=1)
        
        # returning only the 2D goal position without heading
        return self.pos_command_b

    @property
    def path_sampled_ratio(self) -> float:
        """Percentage of the sampled paths that have already been sampled and assigned to a robot"""
        return self.nb_sampled_paths / self.paths.shape[0]

    @property
    def all_path_completed(self) -> torch.Tensor:
        """Check if all the sampled paths have been completed by a robot (i.e. all environments are no longer updated)"""
        return self.not_updated_envs.all()

    @property
    def nb_generated_paths(self) -> int:
        """Number of paths that have been sampled from the environment"""
        return self.paths.shape[0]

    @property
    def analysis(self) -> TerrainAnalysis:
        """The terrain analysis object."""
        return self.traj_sampling.terrain_analyser

    """
    Operations
    """

    def update_trajectory_config(
        self, num_paths: list = [10], min_path_length: list = [0], max_path_length: list = [np.inf]
    ):
        """Update the trajectory configuration for sampling start-goal pairs.

        Args:
            num_paths (list, optional): Number of paths to sample. Defaults to [10].
            min_path_length (list, optional): Minimum path length. Defaults to [0].
            max_path_length (list, optional): Maximum path length. Defaults to [np.inf].
        """
        # Update trajectory config
        self.num_paths = num_paths
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.last_update_config_env_step = self._env.common_step_counter
        self.sample_trajectories()

    def sample_trajectories(self):
        """Sample trajectories"""
        # Sample new start-goal pairs from RRT
        print(
            "[INFO]: Sampling start-goal pairs with following configuration:\n",
            f"\tNumber of paths: {self.num_paths}\n",
            f"\tMinimum path length: {self.min_path_length}\n",
            f"\tMaximum path length: {self.max_path_length}",
        )
        # paths have the entries: [start_x, start_y, start_z, goal_x, goal_y, goal_z, path_length] with shape (num_paths, 7)
        # Note that this module assumes start_z and goal_z are at the robot's base height above the terrain.
        self.paths = self.traj_sampling.sample_paths(
            num_paths=self.num_paths,
            min_path_length=self.min_path_length,
            max_path_length=self.max_path_length,
        )
        print("[INFO]: Sampling has finished.")

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal commands for the specified environments.

        Args:
            env_ids (Sequence[int]): The list of environment IDs to resample.
        """
        # save current state of not updated environments (necessary to log correct information for evaluation)
        self.prev_not_updated_envs = self.not_updated_envs.clone()

        if not self.cfg.infite_sampling:
            # if no infinite sampling, only update for as many environment as there are new trajectories
            if len(env_ids) > self.paths.shape[0] - self.nb_sampled_paths:
                # update non-updated environments
                self.not_updated_envs[env_ids[self.paths.shape[0] - self.nb_sampled_paths :]] = True
                # only update for as many environments as there are new trajectories
                env_ids = env_ids[: max(self.paths.shape[0] - self.nb_sampled_paths, 0)]
            sample = self.paths[self.nb_sampled_paths : self.nb_sampled_paths + len(env_ids)]
            self.nb_sampled_paths += len(env_ids)
        else:
            # Sample new start-goal pairs for terminated environments
            if self.cfg.max_trajectories and len(env_ids) > self.cfg.max_trajectories - self.nb_sampled_paths:
                # update non-updated environments
                self.not_updated_envs[env_ids[self.cfg.max_trajectories - self.nb_sampled_paths :]] = True
                # only update for as many environments as there are new trajectories
                env_ids = env_ids[: self.cfg.max_trajectories - self.nb_sampled_paths]
            self.nb_sampled_paths += len(env_ids)

            sample_idx = torch.randperm(self.paths.shape[0])[: len(env_ids)]
            sample = self.paths[sample_idx]

        # Update command buffers
        self.pos_command_w[env_ids] = sample[:, 3:6].to(self._env.device)

        # Update spawn locations and heading buffer
        self.pos_spawn_w[env_ids] = sample[:, :3].to(self._env.device)
        self.pos_spawn_w[env_ids, 2] += self.cfg.z_offset_spawn

        # Calculate the spawn heading based on the goal position
        self.heading_spawn_w[env_ids] = torch.atan2(
            self.pos_command_w[env_ids, 1] - self.pos_spawn_w[env_ids, 1],
            self.pos_command_w[env_ids, 0] - self.pos_spawn_w[env_ids, 0],
        )
        # Calculate the goal heading based on the goal position
        self.heading_command_w[env_ids] = torch.atan2(
            self.pos_command_w[env_ids, 1] - self.pos_spawn_w[env_ids, 1],
            self.pos_command_w[env_ids, 0] - self.pos_spawn_w[env_ids, 0],
        )

        # Update path length buffer
        self.path_length_command[env_ids] = sample[:, 6].to(self._env.device)

        # NOTE: the reset event is called before the new goal commands are generated, i.e. the spawn locations are
        # updated before the new goal commands are generated. To repsawn with the correct locations, we call here the
        # update spawn locations function
        if self.cfg.reset_pos_term_name:
            reset_term_idx = self._env.event_manager.active_terms["reset"].index(self.cfg.reset_pos_term_name)
            self._env.event_manager._mode_term_cfgs["reset"][reset_term_idx].func(
                self._env, env_ids, **self._env.event_manager._mode_term_cfgs["reset"][reset_term_idx].params
            )

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component
        # self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)[:, :2]
        
        # # resample commands when time is up
        # resampl_ids = torch.
        # goal_dist = torch.norm(target_vec, dim=1)
        # times_up = self.resample_time < 0.1
        # if torch.any(times_up):
        #     self._resample_command(torch.where(times_up)[0])
        #     target_vec[times_up] = self.pos_command_w[times_up] - self.robot.data.root_pos_w[times_up, :3]
        #     target_vec[times_up, 2] = 0.0  # ignore z component

        # update the heading command in the base frame
        # heading_w is angle world x axis to robot base x axis
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        """Update metrics."""
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # init all debug markers common for all goal command generators
        super()._set_debug_vis_impl(debug_vis)

        # create markers if necessary for the first time
        # for each marker type check that the correct command properties exist eg. need spawn position for spawn marker
        if debug_vis:
            if not hasattr(self, "box_spawn_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.box_spawn_visualizer = VisualizationMarkers(marker_cfg)
                self.box_spawn_visualizer.set_visibility(True)
            if not hasattr(self, "goal_heading_visualizer"):
                marker_cfg = CYLINDER_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_heading"
                marker_cfg.markers["cylinder"].height = 1
                marker_cfg.markers["cylinder"].radius = 0.03
                marker_cfg.markers["cylinder"].visual_material.diffuse_color = (0, 0, 1.0)
                self.goal_heading_visualizer = VisualizationMarkers(marker_cfg)
                self.goal_heading_visualizer.set_visibility(True)
        else:
            if hasattr(self, "box_spawn_visualizer"):
                self.box_spawn_visualizer.set_visibility(False)
            if hasattr(self, "heading_goal_visualizer"):
                self.goal_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event, env_ids: Sequence[int] | None = None):
        """Callback function for the debug visualization."""
        if env_ids is None:
            env_ids = slice(None)

        # call the base class debug visualization
        super()._debug_vis_callback(event, env_ids)

        # update spawn marker if it exists
        self.box_spawn_visualizer.visualize(self.pos_spawn_w[env_ids])

        # command heading marker
        orientations = quat_from_euler_xyz(
            torch.zeros_like(self.heading_command_w),
            torch.zeros_like(self.heading_command_w),
            self.heading_command_w,
        )
        translations = self.pos_command_w + quat_rotate(
            orientations, torch.Tensor([0.5, 0, 0]).to(self.device).repeat(orientations.shape[0], 1)
        )
        self.goal_heading_visualizer.visualize(translations[env_ids], orientations[env_ids])
