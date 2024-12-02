# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

from typing import TYPE_CHECKING
from collections.abc import Sequence

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils import configclass, math as math_utils
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg

from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from .sfm_actions_cfg import SFMActionCfg



class SFMAction(ActionTerm):
    """Actions to navigate an obstacle according to the social force model, by following some waypoints"""

    cfg: SFMActionCfg
    _env: ManagerBasedRLEnv
    _asset: RigidObject

    def __init__(self, cfg: SFMActionCfg, env: ManagerBasedRLEnv):
        # TODO check why i can not call super.__init__ throws following error:
        # TypeError: descriptor '__init__' requires a 'super' object but received a 'SFMActionCfg'
        super().__init__(cfg, env)

        # prepare buffers
        self._action_dim = 2  # [vx, vy]
        self._counter = 0
        self._raw_vel_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_vel_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._prev_obstacles_actions_vel = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._obstacles_actions_vel = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._prev_obstacles_actions_pos = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._obstacles_actions_pos = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        if self.cfg.robot_visible:    
            self._robots_positions = self._env.scene.articulations["robot"].data.body_pos_w
        self._sfm_obstacles_positions_w = self._asset.data.body_pos_w

        # self._asset.write_root_pose_to_sim(new_root_pose)

        # set set z position to be above the ground
        # self._sfm_obstacles_positions_w = torch.cat([self._asset.data.body_pos_w[:,:,:2],torch.ones((self.num_envs,1,1), device=self.device)*1.05], dim=2)
        # self._sfm_obstacles_positions_w = self._env.scene.rigid_objects[self.cfg.asset_name].data.body_pos_w
        self._sfm_obstacles_velocity_w = self._asset.data.body_vel_w
        # self._sfm_obstacles_velocity_w = self._env.scene.rigid_objects[self.cfg.asset_name].data.body_vel_w
    


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty()  # self._raw_vel_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_vel_actions
    

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. We run the low-level controller for the obstacles at 50 Hz and therefore we need
        to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:

            self._update_outdated_buffers()
            
            self._prev_obstacles_actions_vel[:] = self._obstacles_actions_vel

            goal_direction = self._get_goal_directions()
            stat_obst_directions = self._get_stat_obstacle_directions()
            # if self.cfg.robot_visible:
            #     robot_directions = self._get_robots_directions()

            total_force = goal_direction - stat_obst_directions 
            normed_total_force = total_force / torch.norm(total_force, p=2, dim=1).unsqueeze(1).expand(self.num_envs, -1)

            self._obstacles_actions_vel[:] = normed_total_force * self.cfg.max_sfm_velocity
            
        self._counter += 1
        

        # when i run it with velocity the obstacle falls down, maybe due to friction, simulation limits?
        
        # Velocity Controller
        # self._asset.write_root_velocity_to_sim(
        #         torch.cat(
        #             [self._obstacles_actions_vel, torch.zeros(self.num_envs, 4, device=self.device)], dim=1
        #             ).to(device=self.device)
        #     )
        
        
        # Position Controller
        self._prev_obstacles_actions_pos = self._obstacles_actions_pos
        self._obstacles_actions_pos = self._obstacles_actions_vel * self._env.step_dt / self.cfg.low_level_decimation
        obstacle_xy_pos = self._asset.data.body_pos_w.squeeze(1)[:, :2] + self._obstacles_actions_pos
        # obstacle_xy_pos = self._sfm_obstacles_positions_w.squeeze(1)[:, :2] + self._obstacles_actions_pos
        
        # fixing the z-koordinate to slightly above ground, otherwise with different command the spawn location is 
        # defined differently w.r.t body origin and some even let the body fall through ground 
        # in general avoids a lot of collision - simulation weird behavior
        obstacle_pos = torch.cat([obstacle_xy_pos, torch.ones((self.num_envs,1), device=self.device)*1.05], dim=1)

        # TODO @vairaviv add heading command if expanded here
        obstacle_quat = math_utils.yaw_quat(self._asset.data.root_quat_w)
        new_root_pose = torch.cat([obstacle_pos, obstacle_quat], dim=1)
        self._asset.write_root_pose_to_sim(new_root_pose)

    """
    Helpers
    """
    def _update_outdated_buffers(self):
        """Update the initialized position and velocity buffers from the simulation"""
        
        self._counter = 0
        if self.cfg.robot_visible:    
            self._agent_position = self._env.scene.articulations["robot"].data.body_pos_w
        self._sfm_obstacles_positions_w = self._asset.data.body_pos_w
        self._sfm_obstacles_velocity_w = self._asset.data.body_vel_w

    def _get_goal_directions(self):
        """Compute the direction for all agents in all envs."""
        target_positions = self._env.command_manager.get_term(self.cfg.command_term_name).pos_command_w[:, :2]
        # target_positions = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)[:,:2]
        current_positions = self._asset.data.root_pos_w[:, :2]

        directions = target_positions-current_positions

        distances = torch.norm(directions, dim=-1, keepdim=True)  
        normalized_directions = torch.where(
            distances > 0, directions / distances, torch.zeros_like(directions)
        ) # linear attraction behavior
        return normalized_directions
    
    def _get_2d_direction(self) -> torch.tensor:
        """Not used anymore as the direction are now directly handled by the ray_caster"""

        # Number of directions
        num_directions = int(
            self._env.scene.sensors[self.cfg.obstacle_sensor].cfg.pattern_cfg.horizontal_fov_range[1] / 
            self._env.scene.sensors[self.cfg.obstacle_sensor].cfg.pattern_cfg.horizontal_res
            )
        # Angles in radians
        angles = torch.linspace(0.0, 2 * math.pi, steps=num_directions, device=self.device)
        # Compute unit vectors for each direction
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1,).to(device=self.device)
        return directions

    def _get_stat_obstacle_directions(self):
        """Compute the direction of static obstacles"""

        # get the ids of the close raycasted mesh points
        lidar_distances = self._env.scene.sensors[self.cfg.obstacle_sensor].data.distances
        mask = lidar_distances < self.cfg.stat_obstacle_radius

        # lidar_directions = self._get_2d_direction().unsqueeze(0).expand(self.num_envs, -1, -1)
        lidar_directions = self._env.scene.sensors[self.cfg.obstacle_sensor].ray_directions_w[:,:,:2]

        filtered_distances = torch.where(mask, lidar_distances, torch.zeros_like(lidar_distances)).to(device=self.device)
        
        # Apply inverse-square scaling for forces: F ~ 1 / (distance^2 + non_zero)
        non_zero = 1e-6  # Small constant to avoid division by zero
        inv_scal_dist = torch.where(
            mask,
            1.0 / (filtered_distances**2 + non_zero),  # Inverse-square scaling
            torch.zeros_like(filtered_distances),
        )

        # Normalize the scaling factors for each environment
        scaling_sum_per_env = inv_scal_dist.sum(dim=1, keepdim=True)
        normalized_scaling = torch.where(
            scaling_sum_per_env > 0,
            inv_scal_dist / scaling_sum_per_env,  # Normalize per environment
            torch.zeros_like(inv_scal_dist),
        )

        # Compute weighted direction vectors
        weighted_directions = lidar_directions * normalized_scaling.unsqueeze(-1)

        # Sum all direction vectors to get the resulting force vector
        resulting_vector = weighted_directions.sum(dim=1)
        
        # distances_sum_per_env = filtered_distances.sum(dim=1)
        # non_zero_distances = distances_sum_per_env > 0.0
        # filtered_dis_normed = torch.where(
        #     non_zero_distances.unsqueeze(dim=1).expand(self.num_envs, -1),
        #     filtered_distances / filtered_distances.sum(dim=1).unsqueeze(dim=1).expand(self.num_envs, -1),
        #     torch.zeros_like(filtered_distances)
        # )

        # normed_direction_vector = lidar_directions * filtered_dis_normed.unsqueeze(-1)

        # resulting_vector = normed_direction_vector.sum(dim=1)
        
        return resulting_vector * 1
    
    def _get_sfm_obstacles_directions(self):
        
        
        raise NotImplementedError
    
    def _get_robots_directions(self):
        # TODO extend it to multiple obstacles in the environment where dim of _robots_positions dont match 
        directions = self._robots_positions - self._asset.data.root_pos_w[:, :2]

        distances = torch.norm(directions, dim=-1, keepdim=True) 
        close_robots = distances < self.cfg.robot_radius 
        normalized_directions = torch.where(
            distances > 0, directions / distances, torch.zeros_like(directions)
        )
        # return normalized_directions
        raise NotImplementedError

    """
    Debug visualizer
    """
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.
        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """

        if debug_vis:
            if not hasattr(self, "action_vector_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Action/action_direction"
                marker_cfg.markers["arrow"].scale = (1.0, 1.0, 1.0)  # scale of the arrow
                marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 0.0, 1.0)  # color set to blue
                self.action_vector_visualizer = VisualizationMarkers(marker_cfg)
                self.action_vector_visualizer.set_visibility(True)
            # if not hasattr(self, "line_to_goal_visualiser"):
            #     marker_cfg = CYLINDER_MARKER_CFG.copy()
            #     marker_cfg.prim_path = "/Visuals/Command/line_to_goal"
            #     marker_cfg.markers["cylinder"].height = 1
            #     marker_cfg.markers["cylinder"].radius = 0.05
            #     self.line_to_goal_visualiser = VisualizationMarkers(marker_cfg)
            #     self.line_to_goal_visualiser.set_visibility(True)
        else:
            if hasattr(self, "action_vector_visualizer"):
                self.action_vector_visualizer.set_visibility(False)
            # if hasattr(self, "line_to_goal_visualiser"):
            #     self.line_to_goal_visualiser.set_visibility(False)

    def _debug_vis_callback(self, event, env_ids: Sequence[int] | None = None):
        """Callback for debug visualization.
        This function calls the visualization objects and sets the data to visualize into them.
        """

        if env_ids is None:
            env_ids = slice(None)

        # update action marker
        self.action_vector_visualizer.visualize(
            translations=self._asset.data.body_pos_w.squeeze(1),
            # orientations=self._obstacles_actions_vel,
        )
