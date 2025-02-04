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

from scipy.spatial import KDTree

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG, CYLINDER_MARKER_CFG
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_euler_xyz,
    wrap_to_pi,
    yaw_quat,
    quat_rotate_inverse,
    quat_from_angle_axis,
)

from nav_collectors.terrain_analysis import TerrainAnalysis

from .goal_command_base import GoalCommandBaseTerm

from crowd_navigation_mt.terrains import SemanticTerrainImporter

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .semantic_terrain_goal_command_cfg import SemanticGoalCommandCfg


class SemanticGoalCommand(GoalCommandBaseTerm):
    r"""Command that generates goal position commands based on terrain and defines the corresponding spawn locations.
    The goal commands are either sampled from RRT or from predefined fixed coordinates defined in the config.

    The goal coordinates/ commands are passed to the planners that generate the actual velocity commands.
    Goal coordinates are sampled in the world frame and then always transformed in the local robot frame.
    """

    cfg: SemanticGoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: SemanticGoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- semantic grid map
        if isinstance(env.scene.terrain, SemanticTerrainImporter):
            # initialized semantic map
            self.grid_map = env.scene.terrain.grid_map
            self.grid_resolution = env.scene.terrain.cfg.semantic_terrain_resolution

            # transform vector in order to shift the gridmap to world frame, scaled for transform
            width = env.scene.terrain.cfg.terrain_generator.size[0]
            height = env.scene.terrain.cfg.terrain_generator.size[1]
            self.transform_vector = torch.tensor([-width / 2, -height / 2]).to(device=self.device) / self.grid_resolution
            
            # valid positions for the robot to spawn 
            self.valid_pos_idx = self.create_valid_positions_idx(
                self.grid_map,
                self.cfg.robot_radius,
                env.scene.terrain.cfg.semantic_terrain_resolution,
            )
            self.valid_pos_w = torch.zeros(self.valid_pos_idx.shape[0], 3, device=self.device)
            self.valid_pos_w[:, :2] = self.convert_idx_to_pos_w(self.valid_pos_idx)
            self.valid_pos_w[:, 2] = 0.8

            self.point_kd_tree = KDTree(self.valid_pos_w[:, :2].cpu())


            # -- goal commands
            self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
            self.pos_command_w = torch.zeros_like(self.pos_command_b)
            self.pos_command_w[:, 2] = 0.8  # offset height, otherwise locomotion policy crashes
            # self.heading_command_b = torch.zeros(self.num_envs, device=self.device)
            # self.heading_command_w = torch.zeros_like(self.heading_command_b)

            # -- spawn location
            self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
            self.pos_spawn_w[:, 2] = 0.8  # offset height, otherwise locomotion policy crashes
            self.pos_spawn_w[:, :2] = self.valid_pos_w[:self.num_envs, :2]
            self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)

            # radius for increasing spawn radius 
            self.radius_lvl = torch.ones(self.num_envs, device=self.device) * self.cfg.sampling_radius

        # if semantic grid map not available
        else:
            print("[INFO]: Semantic Grid Map is not available for spawn positions and goal commands.")
            
            self.grid_map = None
            # -- goal commands
            self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
            self.pos_command_w = torch.zeros_like(self.pos_command_b)
            self.pos_command_w[:, 2] = 0.8  # offset height, otherwise locomotion policy crashes
            # self.heading_command_b = torch.zeros(self.num_envs, device=self.device)
            # self.heading_command_w = torch.zeros_like(self.heading_command_b)

            # -- spawn location
            # TODO check assumption: env frame and global frame have same z (xy planes align)
            # self.pos_spawn_w = self.terrain_analysis.sample_spawn(env.scene.env_origins[:, :2])
            self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
            self.pos_spawn_w[:, 2] = 0.8  # offset height, otherwise locomotion policy crashes
            self.pos_spawn_w[:, :2] = env.scene.env_origins[:, :2]
            self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)
               
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["N_successes"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vel_to_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vel_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["increments"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["x_pos"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["y_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.success_rate_buffer = torch.zeros(self.num_envs, 10, device=self.device)

        # helpers for metrics
        self.goal_reached_counter = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "SemanticGoalCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        # return torch.cat((self.pos_command_b[:, :2], self.heading_command_b.unsqueeze(1)), dim=1)
        return self.pos_command_b[:, :2]

    """
    Implementation specific functions.
    """

    def _resample_spawn_positions(self, env_ids: Sequence[int]):
        if self.grid_map is None:
            self.pos_spawn_w[env_ids, :2] = self._env.scene.env_origins[env_ids, :2]
        else:
            # idx = self.point_kd_tree.query_ball_point(self.pos_command_w[env_ids, :2].cpu(), r=self.radius_lvl[env_ids].cpu())
            # #random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
            # random_idx = torch.tensor([list[-1] for list in idx])
            # self.pos_spawn_w[env_ids, :2] = self.valid_pos_w[random_idx, :2] 

            # new_goal_distance = torch.norm((self.pos_command_w-self.pos_spawn_w), dim=-1)  

            # get valid spawn locations from the command
            random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
            positions = self.valid_pos_w[random_idx, :]

            # overwrite spawn position in goal command for other calculations
            self.pos_spawn_w[env_ids] = positions    


    def _resample_command(self, env_ids: Sequence[int]):
        """sample new goal positions.
        It is sampled on a sidewalk and randomly selected from the valid positions."""

        if self.grid_map is None:
            self.pos_command_w[env_ids, :2] = self._env.scene.env_origins[env_ids, :2]
        # else:
        #     random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
        #     self.pos_command_w[env_ids, :2] = self.valid_pos_w[random_idx, :2]

        # else:  
        #     self._resample_spawn_positions(env_ids)
        #     idx = self.point_kd_tree.query_ball_point(self.pos_spawn_w[env_ids, :2].cpu(), r=self.radius_lvl[env_ids].cpu())
        #     #random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
        #     try:
        #         # random_idx = torch.tensor([list[-1] for list in idx])
        #         random_idx = torch.tensor(
        #             [
        #                 list[torch.randint(0, len(list), (1,)).item()] 
        #                 for list in idx
        #             ]
        #         )
        #     except IndexError:
        #         print("[DEBUG]: the root position of the robot is at an invalid position")
        #         random_idx = torch.randint(0, self.valid_pos_w.shape[0], (len(env_ids),))
        #     self.pos_command_w[env_ids, :2] = self.valid_pos_w[random_idx, :2] 

        else: 
            self._resample_spawn_positions(env_ids)
            idx = self.point_kd_tree.query_ball_point(self.pos_spawn_w[env_ids, :2].cpu(), r=self.radius_lvl[env_ids].cpu())
            # self.pos_spawn_w[env_ids, :2] = self.robot.data.root_pos_w[env_ids, :2]
            #random_idx = torch.randint(0, self.valid_pos_idx.size(0), (len(env_ids),), device=self.device)
            try:
                random_idx = torch.tensor(
                    [
                        list[torch.randint(0, len(list), (1,)).item()] 
                        if len(list) > 0 
                        else (
                            print(f"[INFO]:  SemanticGoalCommand: Empty list encountered, using self.valid_pos_w instead") or
                            torch.randint(0, self.valid_pos_w.shape[0], (1,)).item()
                        ) # evalutates both and easier for debugging
                        for list in idx
                    ]
                ).to(device=self.device)
                # if self.pos_spawn_w[]
            except IndexError:
                print("[DEBUG]: SemanticGoalCommand: the root position of the robot is at an invalid position")
                random_idx = torch.randint(0, self.valid_pos_w.shape[0], (len(env_ids),))
            
            if torch.any(
                torch.isclose(self.valid_pos_w[random_idx, 0], self.pos_spawn_w[env_ids, 0], 1e-4) & 
                torch.isclose(self.valid_pos_w[random_idx, 1], self.pos_spawn_w[env_ids, 1], 1e-4)
            ):
                mask = (
                    torch.isclose(self.valid_pos_w[random_idx, 0], self.pos_spawn_w[env_ids, 0], 1e-4) & 
                    torch.isclose(self.valid_pos_w[random_idx, 1], self.pos_spawn_w[env_ids, 1], 1e-4)
                ).to(device=self.device)
                if torch.any(random_idx[mask] == 0):
                    print("[DEBUG]: SemanticGoalCommand: The random idx is 0!")
                random_idx[mask] -= 1  # this will never be out of index as list[-1] is valid too, and else it samples from 1 to valid_pos_w.shape[0]
                print("[DEBUG]: SemanticGoalCommand: Position Command and Spawn location are the same!")
            self.pos_command_w[env_ids, :2] = self.valid_pos_w[random_idx, :2] 

        failure = (
            self._env.termination_manager.terminated[env_ids]
            & ~self._env.termination_manager._term_dones["goal_reached"][env_ids]
        )
        self.goal_reached_counter[env_ids] -= failure.int()
        self.goal_reached_counter[env_ids] += (~failure).int()

        # # resample start position
        # self._resample_spawn_positions(env_ids)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        # self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        # self.heading_command_b[:] = torch.atan2(self.pos_command_b[:, 1], self.pos_command_b[:, 0])
        # self.quat_command_b[:] = quat_from_euler_xyz(
        #     torch.zeros((self.heading_command_b.shape), device=self.device),
        #     torch.zeros((self.heading_command_b.shape), device=self.device),
        #     self.heading_command_b,
        # )
        # if self.visualize_plot:
        #     self.plot._plot(self)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "line_to_goal_visualiser"):
                marker_cfg = CYLINDER_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/line_to_goal"
                marker_cfg.markers["cylinder"].height = 1
                marker_cfg.markers["cylinder"].radius = 0.05
                self.line_to_goal_visualiser = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "body_visualizer") and False:
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/body"
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 1.0, 1.0)
                self.body_visualizer = VisualizationMarkers(marker_cfg)
                marker_cfg.prim_path = "/Visuals/Command/root"
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 0.0)
                self.root_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.box_goal_visualizer.set_visibility(True)
            if self.cfg.robot_to_goal_line_vis:
                self.line_to_goal_visualiser.set_visibility(True)
            # self.body_visualizer.set_visibility(True)
            # self.root_visualizer.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if hasattr(self, "line_to_goal_visualiser"):
                self.line_to_goal_visualiser.set_visibility(False)
            # if hasattr(self, "body_visualizer"):
            #     self.body_visualizer.set_visibility(False)
            # if hasattr(self, "root_visualizer"):
            #     self.root_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback function for the debug visualization."""
        # update the box marker
        self.box_goal_visualizer.visualize(self.pos_command_w)
        # Update the body and root visualizer
        # self.body_visualizer.visualize(
        #     self.robot.data.body_pos_w[:, 0, :3] + torch.Tensor([0, 0, 1.0]).to(self.robot.data.body_pos_w.device)
        # )
        # self.root_visualizer.visualize(
        #     self.robot.data.body_pos_w[:, 0, :3] + torch.Tensor([0, 0, 2.0]).to(self.robot.data.body_pos_w.device)
        # )
        # update the line marker
        # calculate the difference vector between the robot root position and the goal position
        difference = self.pos_command_w - self.robot.data.root_pos_w  # self.robot.data.body_pos_w[:, 0, :3]
        translations = self.robot.data.root_pos_w.clone()  # self.robot.data.body_pos_w[:, 0, :3]
        # calculate the scale of the arrow (Mx3)
        scales = torch.norm(difference, dim=1)
        # translate half of the length along difference axis
        translations += difference / 2
        # scale along x axis
        scales = torch.vstack([scales, torch.ones_like(scales), torch.ones_like(scales)]).T
        # convert the difference vector to a quaternion
        difference = torch.nn.functional.normalize(difference, dim=1)
        x_vec = torch.tensor([1, 0, 0]).float().to(self.pos_command_w.device)
        angle = -torch.acos(difference @ x_vec)
        axis = torch.linalg.cross(difference, x_vec.expand_as(difference))
        quat = quat_from_angle_axis(angle, axis)
        # apply transforms
        if self.cfg.robot_to_goal_line_vis:
            self.line_to_goal_visualiser.visualize(translations=translations, scales=scales, orientations=quat)

    def _update_metrics(self):
        """Update metrics."""
        pos_error = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.metrics["error_pos"] = torch.norm(pos_error[:, :2], dim=1)
        # self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
        self.metrics["N_successes"] = self.goal_reached_counter.clone()

        vel = self.robot.data.root_state_w[:, 7:9]
        self.metrics["vel_abs"] = torch.norm(vel, dim=1)
        pos_error_dir = pos_error[:, :2] / (torch.norm(pos_error[:, :2], dim=1, keepdim=True) + 1e-6)
        self.metrics["vel_to_goal"] = (vel * pos_error_dir).sum(dim=1)

        # self.metrics["increments"] = self.goal_dist_increment.clone()

        # self.metrics["x_pos"] = self.robot.data.root_pos_w[:, 0]
        # self.metrics["y_pos"] = self.robot.data.root_pos_w[:, 1]

        self.metrics["success_rate"] = self.success_rate_buffer.mean(dim=1) / 2 + 0.5

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        failed = self._env.termination_manager._term_dones["illegal_contact"][env_ids]
        succeded = self._env.termination_manager._term_dones["goal_reached"][env_ids]
        # failed = failed & ~succeded

        self.success_rate_buffer[env_ids] = torch.roll(self.success_rate_buffer[env_ids], 1, dims=1)
        self.success_rate_buffer[env_ids, 0] = succeded.float() - failed.float()

        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # # resample the command
        # this is already done in the reset robot position event and thus not needed here
        # self._resample(env_ids)
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras

    """
    Helpers
    """

    def create_valid_positions_idx(self, grid_map: torch.tensor, robot_radius: float, grid_resolution: float):
        """The Robot should be spawned on a sidewalk and should have a buffer to the streets and other obstacles"""
        
        # buffer for robot radius
        robot_buffer_cells = math.ceil(robot_radius / grid_resolution)

        # prep for 2D convolution
        kernel_size = 2 * robot_buffer_cells + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32, device=self.device)
        grid_map_prepped = grid_map.unsqueeze(0).unsqueeze(0).float()

        grid_mask = (grid_map_prepped > 0.0)

        grid_mask_padded = torch.nn.functional.pad(
            grid_mask,
            (robot_buffer_cells, robot_buffer_cells, robot_buffer_cells, robot_buffer_cells),
            mode="constant",
            value=True,
        )
        grid_buffered = torch.nn.functional.conv2d(
            grid_mask_padded.float(), 
            kernel.unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)

        valid_mask = (grid_buffered == 0.0)
        valid_positions_idx = torch.argwhere(valid_mask)

        return valid_positions_idx

    def convert_idx_to_pos_w(self, idx: torch.tensor):
        
        transformed_idx = idx + self.transform_vector 
        pos_w_2D = self.grid_resolution * transformed_idx

        return pos_w_2D

    """
    Operations, used for curriculum learning
    """

    def update_goal_distance(self, increase_by: float | None = None, required_successes: int = 1):
        """Update the goal distance."""
        increase_goal_dist = self.goal_reached_counter >= required_successes
        decrease_goal_dist = self.goal_reached_counter < -required_successes

        self.goal_reached_counter[increase_goal_dist | decrease_goal_dist] = 0

        self.goal_dist[increase_goal_dist] += increase_by
        self.goal_dist[decrease_goal_dist] -= increase_by
        self.goal_dist = torch.clamp(self.goal_dist, min=1, max=self.cfg.max_goal_distance)
        # TODO check why radius and max_goal_distance is used in the same context
        # self.goal_dist = torch.clamp(self.goal_dist, min=1, max=self.cfg.radius)

        # success_rate = self.goal_reached_counter.float().mean()

        # if success_rate >= required_successes:
        #     # reset the counter
        #     self.goal_reached_counter = torch.zeros_like(self.goal_reached_counter)
        #     if increase_by is not None:
        #         self.goal_dist += increase_by
        #     elif new_goal_dist is not None:
        #         self.goal_dist = new_goal_dist
        #     else:
        #         raise ValueError("Either new_goal_dist or increase_by must be provided.")

        #     self.goal_dist = min(self.goal_dist, self.cfg.max_goal_distance)

    def update_goal_distance_rel(self, increase_by: int = 1, required_successes: int = 1, max_goal_dist: int = 10):
        """Update the goal distance relative."""
        increase_goal_dist = self.goal_reached_counter >= required_successes
        decrease_goal_dist = self.goal_reached_counter < -required_successes

        self.goal_reached_counter[increase_goal_dist | decrease_goal_dist] = 0

        self.goal_dist_rel[increase_goal_dist] += increase_by
        self.goal_dist_rel[decrease_goal_dist] -= increase_by
        self.goal_dist_rel = torch.clamp(self.goal_dist_rel, min=1, max=self.cfg.max_goal_distance)
        # TODO check why radius and max_goal_distance is used in the same context
        # self.goal_dist_rel = torch.clamp(self.goal_dist_rel, min=1, max=self.cfg.radius)
