# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass

from .robot_lvl_goal_command import RobotLvlGoalCommand

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg


"""
Base command generator.
"""
from omni.isaac.lab.envs.mdp.commands import UniformPose2dCommandCfg

from nav_collectors.terrain_analysis import TerrainAnalysis, TerrainAnalysisCfg

@configclass
class RobotLvlGoalCommandCfg(CommandTermCfg):
    """Configuration for the robot goal command generator."""

    class_type: type = RobotLvlGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    radius: float = 1
    """Radius of the goal area around the robot."""

    max_goal_distance: float = 25
    """Maximum distance of the goal from the robot."""

    terrain_analysis: TerrainAnalysisCfg | None = None
    """Terrain analysis configuration."""

    angles: list[float] | None = None
    """List of angles in radians to sample from. If None, the angles are sampled uniformly from 0 to 2pi."""

    use_grid_spacing: bool = True
    """If true, the spacing between the terrain cells is used to determine the range of the area.
    Usesfull for grid based terrains with clear spawning area in the middle."""

    deterministic_goal: bool = False
    """If true, the goal is always forward by the same amount."""

    deterministic_goal_distance_x: float = 3.0
    """Distance of the deterministic goal from the robot x."""

    deterministic_goal_distance_y: float = 3.0
    """Distance of the deterministic goal from the robot y."""

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""