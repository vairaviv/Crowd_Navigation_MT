from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
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
from omni.isaac.lab.envs.mdp.commands import UniformPose2dCommandCfg, UniformPose2dCommand
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from .robot_lvl_goal_command_cfg import RobotLvlGoalCommandCfg

from nav_collectors.terrain_analysis import TerrainAnalysis, TerrainAnalysisCfg


"""Robot pose command"""


class RobotLvlGoalCommand(CommandTerm):
    """Command generator for generating pose commands for the robot."""

    cfg: RobotLvlGoalCommandCfg
    rewards: dict = dict()

    def __init__(self, cfg: RobotLvlGoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # -- terrain
        self.terrain = env.scene.terrain
        self.levels = self.terrain.terrain_levels
        self.types = self.terrain.terrain_types

        # -- goal commands
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_w = torch.zeros_like(self.pos_command_b)
        self.pos_command_w[:, 2] = 0.8

        # -- run terrain analysis
        if self.cfg.terrain_analysis:
            self.terrain_analysis = TerrainAnalysis(cfg=self.cfg.terrain_analysis, scene=self._env.scene)
            self.terrain_analysis.analyse()

            # -- fit kd-tree based on terrain level on all the graph nodes to quickly find the closest node to the robot
            pruning_tensor = torch.ones(self.terrain_analysis.points.shape[0], dtype=bool, device=self.device)
            pruning_tensor[self.terrain_analysis.isolated_points_ids] = False

            # -- Get grid dimensions
            self.num_levels, self.num_types, _ = env.scene.terrain.terrain_origins.shape

            # Expand terrain origins to [num_levels * num_types, 3] for computation
            terrain_origins_flat = env.scene.terrain.terrain_origins.view(-1, 3)  # Shape: [num_levels * num_types, 3]

            # terrain bounding box for each level and typ:
            self.terrain_bb = torch.cat(
                (terrain_origins_flat[:,:2] - torch.tensor(env.scene.terrain.cfg.terrain_generator.size).to(self.device)/2,
                 terrain_origins_flat[:,:2] + torch.tensor(env.scene.terrain.cfg.terrain_generator.size).to(self.device)/2),
                dim=1
            )

            # Compute distances between points and terrain origins
            points_expanded = self.terrain_analysis.points.unsqueeze(1)  # Shape: [num_points, 1, 3]
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
                    points_in_cell = self.terrain_analysis.points[terrain_mask]  # Extract points for the cell
                    # Concatenate the grouped points
                    self.grouped_points[level].append(points_in_cell)

            # To look up close points in the terrain, not used currently but might be useful for later
            self._grouped_kd_trees = []
            for level in range(self.num_levels):
                self._grouped_kd_trees.append([])
                for type in range(self.num_types):
                    self._grouped_kd_trees[level].append(KDTree(self.grouped_points[level][type].cpu().numpy()))

        # self.heading_command_b = torch.zeros(self.num_envs, device=self.device)
        # self.heading_command_w = torch.zeros_like(self.heading_command_b)

        # -- spawn location
        self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
        

        # TODO check assumption: env frame and global frame have same z (xy planes align)
        if self.cfg.terrain_analysis:
            # TODO: @vairaviv need to fix this
            # self.pos_spawn_w = self.grouped_points[self.levels][self.types][] 
            # self.pos_spawn_w = self.terrain_analysis.sample_spawn(env.scene.env_origins[:, :2])
            self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_spawn_w[:, 2] = 0.8
        self.pos_spawn_w[:, :2] = env.scene.env_origins[:, :2]

        self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)

        # -- metrics
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
        
        # -- rewards
        self.rewards["goal_heading"] = torch.zeros(self.num_envs, device=self.device)

        # -- environment
        self.env = env

        # -- goal reached counter
        self.goal_reached_counter = torch.zeros(self.num_envs, device=self.device)

        # -- goal distance
        self.goal_dist = cfg.radius * torch.ones(env.num_envs, device=self.device)
        self.goal_dist_rel = torch.ones(env.num_envs, device=self.device)
        self.goal_dist_increment = torch.ones(env.num_envs, device=self.device)
        self.use_grid_spacing = cfg.use_grid_spacing

        self.angles = torch.tensor(cfg.angles, device=self.device) if cfg.angles is not None else None

        # for plotting the commands 
        # TODO get the terrain limits from env
        self.visualize_plot = False
        terrain_x_lim = (-10, 10)
        terrain_y_lim = (-10, 10)
        self.plot = RobotGoalCommandPlot(terrain_x_lim, terrain_y_lim)

    def __str__(self) -> str:
        msg = "GoalCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tGoal distance: {tuple(self.goal_dist.mean())}\n"
        # msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Spawn location
    """

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
        if self.terrain_analysis is not None:
            if len(env_ids) > 0:
                self.pos_spawn_w[env_ids] = self.terrain_analysis.sample_spawn(self.env.scene.env_origins[env_ids, :2])
                # TODO randomize spawnpositions currently taken out as raycast_dynamic_meshes doesnt work
                # self.pos_spawn_w[env_ids] = self.pos_spawn_w[env_ids]
        else:
            self.pos_spawn_w[env_ids, :2] = self.env.scene.env_origins[env_ids, :2]

    def _resample_command(self, env_ids: Sequence[int]):
        """sample new goal positions and headings.
        The goal positions are sampled uniformly in a circle around the robot within a given rectangle.
        Headings are sampled uniformly."""

        # update goals reached counter, happens in termination condition
        # goal_reached = self.env.observation_manager.compute_group(group_name="metrics")["goal_reached"]
        # self.goal_reached_counter[env_ids] += goal_reached.squeeze()[env_ids].int()

        # # # # sample new goal positions and headings in world frame aligned body frame (origin at base, axis aligned with world)
        # sample in world frame!!!

        # random angle 90 deg spacing

        if self.use_grid_spacing:
            # try:
            #     terrain_spacing = self.env.scene.terrain.cfg.terrain_generator.size[0]
            # except AttributeError:
            #     terrain_spacing = 1
            if self.env.scene.terrain.cfg.terrain_type == "generator":
                terrain_spacing = self.env.scene.terrain.cfg.terrain_generator.size[0]
            else:
                terrain_spacing = 1
            max_rel_spacing = self.goal_dist_rel[env_ids] * self.goal_dist_increment[env_ids]

            rand_floats_x = torch.rand_like(env_ids.float(), device=self.device)
            rand_int_spacing_x = (rand_floats_x * (max_rel_spacing + 1)).int()

            rand_floats_y = torch.rand_like(env_ids.float(), device=self.device)
            rand_int_spacing_y = (rand_floats_y * (max_rel_spacing + 1)).int()

            # prevent both from being zero
            both_zero = (rand_int_spacing_x == 0) & (rand_int_spacing_y == 0)
            set_x_to_one = torch.rand(rand_int_spacing_x.shape, device=rand_int_spacing_x.device) < 0.5
            set_x_to_one = set_x_to_one & both_zero  # Apply both_zero mask
            rand_int_spacing_x[set_x_to_one] += 1
            rand_int_spacing_y[both_zero & ~set_x_to_one] = 1

            rand_sign_x = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1
            rand_sign_y = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1

            if self.cfg.deterministic_goal:
                rand_int_spacing_x = torch.ones_like(rand_int_spacing_x) * self.cfg.deterministic_goal_distance_x
                rand_int_spacing_y = torch.ones_like(rand_int_spacing_y) * self.cfg.deterministic_goal_distance_y
                rand_sign_x = torch.ones_like(rand_sign_x)
                rand_sign_y = torch.ones_like(rand_sign_y)

            x_pos_w = terrain_spacing * rand_int_spacing_x * rand_sign_x
            y_pos_w = terrain_spacing * rand_int_spacing_y * rand_sign_y

        else:
            if self.angles is None:
                random_angle = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
            else:
                angles = self.angles
                random_indices = torch.randint(0, len(angles), (len(env_ids),), device=self.device)
                random_angle = angles[random_indices]
            # try:
            #     terrain_spacing = self.env.scene.terrain.cfg.terrain_generator.size[0]
            # except AttributeError:
            #     terrain_spacing = 1

            if self.env.scene.terrain.cfg.terrain_type == "generator":
                terrain_spacing = self.env.scene.terrain.cfg.terrain_generator.size[0]
            else:
                terrain_spacing = 2

            # x_pos_b = self.goal_dist[env_ids] * torch.cos(random_angle)
            # y_pos_b = self.goal_dist[env_ids] * torch.sin(random_angle)
            x_pos_w = (
                terrain_spacing
                * torch.cos(random_angle)
                * self.goal_dist_rel[env_ids]
                * self.goal_dist_increment[env_ids]
            )
            y_pos_w = (
                terrain_spacing
                * torch.sin(random_angle)
                * self.goal_dist_rel[env_ids]
                * self.goal_dist_increment[env_ids]
            )
        self.pos_command_w[env_ids, 0] = self.env.scene.env_origins[env_ids, 0] + x_pos_w
        self.pos_command_w[env_ids, 1] = self.env.scene.env_origins[env_ids, 1] + y_pos_w

        self.pos_command_w = self._clamp_to_area(self.pos_command_w)

        # self.pos_command_w[env_ids] = self.robot.data.root_pos_w[env_ids, :3] + self.pos_command_b[env_ids]
        # round position to the nearest terrain origin and set
        # clamp the goal positions to the environment boundaries

        # random_heading = torch.rand(len(env_ids), device=self.device) * 2 * math.pi - math.pi
        # self.heading_command_w[env_ids] = random_heading

        failure = (
            self.env.termination_manager.terminated[env_ids]
            & ~self.env.termination_manager._term_dones["goal_reached"][env_ids]
        )
        self.goal_reached_counter[env_ids] -= failure.int()
        self.goal_reached_counter[env_ids] += (~failure).int()

        # resample start position
        self._resample_spawn_positions(env_ids)

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
        if self.visualize_plot:
            self.plot._plot(self)

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

        self.metrics["increments"] = self.goal_dist_increment.clone()

        self.metrics["x_pos"] = self.robot.data.root_pos_w[:, 0]
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
        failed = self.env.termination_manager._term_dones["illegal_contact"][env_ids]
        succeded = self.env.termination_manager._term_dones["goal_reached"][env_ids]
        # failed = failed & ~succeded

        self.success_rate_buffer[env_ids] = torch.roll(self.success_rate_buffer[env_ids], 1, dims=1)
        self.success_rate_buffer[env_ids, 0] = succeded.float() - failed.float()

        self.goal_dist_increment[env_ids] = 1

        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self._resample(env_ids)
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

    def increment_goal_distance(self, env_ids: Sequence[int]):
        self.goal_dist_increment[env_ids] += 1

    def _is_in_area(self, pos: torch.Tensor) -> torch.Tensor:
        """Check if a point is in the arena."""
        # TODO check if correct
        square_arena_side_length = math.ceil(self.num_envs**0.5) * self.env.cfg.scene.env_spacing
        return (pos.abs() < square_arena_side_length / 2).all(dim=1)

    def _clamp_to_area(self, pos: torch.Tensor) -> torch.Tensor:
        """Clamp a point to the arena."""
        if self.env.scene.terrain.cfg.terrain_type == "generator":

            max_x, max_y = self.env.scene.terrain.terrain_origins.view(-1, 3).max(dim=0)[0][:2]
            min_x, min_y = self.env.scene.terrain.terrain_origins.view(-1, 3).min(dim=0)[0][:2]

            # square_arena_side_length = math.ceil(self.num_envs**0.5) * self.env.cfg.scene.env_spacing
            # pos[:, :2] = torch.clamp(pos[:, :2], -square_arena_side_length / 2, square_arena_side_length / 2)
            pos[:, 0] = torch.clamp(pos[:, 0], min_x, max_x)
            pos[:, 1] = torch.clamp(pos[:, 1], min_y, max_y)

        return pos

    def update_success(self, at_goal: torch.Tensor):
        """Update the goal reached counter."""
        self.goal_reached_counter += at_goal.int()

    def update_failures(self, failed: torch.Tensor):
        """Update the goal reached counter."""
        self.goal_reached_counter -= failed.int()

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
