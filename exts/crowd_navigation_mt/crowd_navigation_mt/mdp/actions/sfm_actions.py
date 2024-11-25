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

if TYPE_CHECKING:
    from .sfm_actions_cfg import SFMActionCfg


class SFMAction(ActionTerm):
    """Actions to navigate an obstacle according to the social force model, by following some waypoints"""

    cfg: SFMActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SFMActionCfg, env: ManagerBasedRLEnv):
        # TODO check why i can not call super.__init__ throws following error:
        # TypeError: descriptor '__init__' requires a 'super' object but received a 'SFMActionCfg'
        super.__init__(cfg,env)

        # prepare buffers
        self._action_dim = 3  # [Fx, Fy, Fz]
        self._counter = 0
        self._raw_force_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_force_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._prev_obstacles_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._obstacle_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        if self.cfg.robot_visible:    
            self._agent_position = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._sfm_obstacle_positions_w = self._env.scene.rigid_objects[self.cfg.asset_name].data.body_pos_w
        # self._sfm_obstacle_positions_w = self._asset.data.body_pos_w
        self._sfm_obstacle_velocity_w = self._env.scene.rigid_objects[self.cfg.asset_name].data.body_vel_w


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_force_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_force_actions
    

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
            self._counter = 0
            self._prev_obstacles_actions[:] = self._obstacle_actions

            # process actions and bring them in the right order
            self._obstacle_actions[:] = self._env.scene.rigid_objects["sfm_obstacle"].data.root_lin_vel_w
            

            

        
        self._counter += 1

        