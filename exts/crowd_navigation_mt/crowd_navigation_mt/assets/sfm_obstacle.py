# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils

from omni.isaac.lab.assets import AssetBase, AssetBaseCfg
from omni.isaac.lab.assets.rigid_object import RigidObjectData, RigidObject, RigidObjectCfg

if TYPE_CHECKING:
    from .sfm_obstacle_cfg import SFMObstacleCfg


class SFMObstacle(RigidObject):

    def __init__(self, cfg: SFMObstacleCfg):
        super().__init__(cfg)
        self._device = sim_utils.SimulationContext.instance().device

        # terrain_level and terrain_type given as tuple 
        start_id = 0
        tot_num_agent = int(cfg.num_sfm_agent_increase * cfg.num_levels * (cfg.num_levels + 1) / 2 * cfg.num_types)
        self.lvl_typ = torch.zeros(tot_num_agent, 2, device=self._device)
        for level in range(4):
            for _type in range(4):
                end_id = start_id + (_type + 1) * cfg.num_sfm_agent_increase
                
                # if end_id > self.num_envs:
                #     print("Not enough env spawned to fill up the whole terrain!")
                #     end_id = start_id
                #     break

                self.lvl_typ[start_id:end_id, :] = torch.tensor([level, _type])
                start_id = end_id

        # if end_id < cfg.num_levels * (cfg.num_types * (cfg.num_types + 1) / 2):
        #     print("Not enough env spawned to fill up the whole terrain!")

        # TODO: @ vairaviv how to get the total amount of num_envs at this stage?
        # elif end_id < self.num_envs:
        #     print("Too many dynamic obstacles, they will be spawned below the plane!")
        #     # TODO: @ vairaviv just did it ugly if time maybe store them some where 
        #     obstacle_pos[end_id:, :] = obstacle_pos[end_id:, :] * -2


