# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg

from .sfm_obstacle_importer import SFMObstacleImporter

from omni.isaac.lab.terrains import TerrainImporterCfg

@configclass
class SFMObstacleImporterCfg(TerrainImporterCfg):

    class_type: type = SFMObstacleImporter
    """The class to use for the social force model importer.
    
    Defaults to :class:`crowd_navigation_mt.terrains.importer.SFMObstacleImporter`
    """

    num_sfm_obstacles: int = 1
    """The amount of dynamic obstacles wanted for the environment"""

    prim_path = MISSING
    """the default prim path for social force model obstacles"""

    collision_group: int = -1
    """The collision group of the dynamic obstacle. 
    
    Defaults to -1, implicating global collisions.
    """
