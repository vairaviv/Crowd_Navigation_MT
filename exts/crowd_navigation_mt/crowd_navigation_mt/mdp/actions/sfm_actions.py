# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg

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
        self._prev_obstacles_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._obstacles_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        if self.cfg.robot_visible:    
            self._robots_positions = self._env.scene.articulations["robot"].data.body_pos_w
        self._sfm_obstacles_positions_w = self._asset.data.body_pos_w
        # self._sfm_obstacle_positions_w = self._env.scene.rigid_objects[self.cfg.asset_name].data.body_pos_w
        self._sfm_obstacles_velocity_w = self._asset.data.body_vel_w
        # self._sfm_obstacle_velocity_w = self._env.scene.rigid_objects[self.cfg.asset_name].data.body_vel_w
    


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_vel_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_vel_actions
    

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        return None

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. We run the low-level controller for the obstacles at 50 Hz and therefore we need
        to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:

            self._update_outdated_buffers()
            
            self._prev_obstacles_actions[:] = self._obstacles_actions

            goal_direction = self._get_goal_directions()
            # stat_obst_directions = self._get_stat_obstacle_directions()
            # if self.cfg.robot_visible:
            #     robot_directions = self._get_robots_directions()

            total_force = goal_direction
            # process actions and bring them in the right order
            # self._obstacles_actions[:] = self._asset.data.root_lin_vel_w[:,:2]


            self._obstacles_actions[:] = total_force * self.cfg.max_sfm_velocity
            self._asset.write_root_velocity_to_sim(
                torch.cat(
                    [self._obstacles_actions, torch.zeros(self.num_envs, 4, device=self.device)], dim=1
                    ).to(device=self.device)
            )
        self._counter += 1
        

    """
    Helpers
    """
    def _update_outdated_buffers(self):
        """Update the initialized position and velocity buffers from the simulation"""
        
        self._counter = 0
        if self.cfg.robot_visible:    
            self._agent_position = self._env.scene.articulations["robot"].data.body_pos_w
        self._sfm_obstacle_positions_w = self._asset.data.body_pos_w
        self._sfm_obstacle_velocity_w = self._asset.data.body_vel_w

    def _get_goal_directions(self):
        """Compute the direction for all agents in all envs."""

        target_positions = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)[:,:2]
        current_positions = self._asset.data.root_pos_w[:, :2]

        directions = target_positions-current_positions

        distances = torch.norm(directions, dim=-1, keepdim=True)  
        normalized_directions = torch.where(
            distances > 0, directions / distances, torch.zeros_like(directions)
        ) # linear attraction behavior
        return normalized_directions
    
    def _get_stat_obstacle_directions(self):
        """Compute the direction of static obstacles"""

        # get the ids of the close raycasted mesh points
        lidar_distances = self._env.scene.sensors[self.cfg.obstacle_sensor].data.distances
        mask = lidar_distances < self.cfg.stat_obstacle_radius

        # ids of the rays
        ray_id = torch.nonzero(mask).flatten

        # get the coordinate of the close points
        stat_obstacle_point_w = self._env.scene.sensors[self.cfg.obstacle_sensor].data.ray_hits_w[mask]
        # stat_obstacle_point_w = self._env.scene.sensors[self.cfg.obstacle_sensor].data.ray_hits_w[ray_id]

        current_positions = self._asset.data.root_pos_w[:, :2]

        return NotImplementedError
    
    def _get_sfm_obstacles_directions(self):
        
        
        return NotImplementedError
    
    def _get_robots_directions(self):
        # TODO extend it to multiple obstacles in the environment where dim of _robots_positions dont match 
        directions = self._robots_positions - self._asset.data.root_pos_w[:, :2]

        distances = torch.norm(directions, dim=-1, keepdim=True) 
        close_robots = distances < self.cfg.robot_radius 
        normalized_directions = torch.where(
            distances > 0, directions / distances, torch.zeros_like(directions)
        )
        # return normalized_directions
        return NotImplementedError

