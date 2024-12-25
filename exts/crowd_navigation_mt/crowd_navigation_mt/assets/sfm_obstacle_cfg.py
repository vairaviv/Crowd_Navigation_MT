# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object import RigidObject, RigidObjectCfg

from .sfm_obstacle import SFMObstacle

@configclass
class SFMObstacleCfg(RigidObjectCfg):
    """Configuration parameters for a rigid object."""

    class_type: type = SFMObstacle

    num_sfm_agent_increase: int = 5
    "Used to increase the density of the dynamic obstacle in the environment"

    num_levels : int = MISSING
    "number of terrain level"

    num_types : int = MISSING
    "number of terrain types"