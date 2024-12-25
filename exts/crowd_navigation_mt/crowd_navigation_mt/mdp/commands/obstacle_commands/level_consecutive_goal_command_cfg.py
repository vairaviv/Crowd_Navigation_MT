# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from nav_collectors.terrain_analysis import TerrainAnalysisCfg

from .level_consecutive_goal_command import LvlConsecutiveGoalCommand
from .goal_command_base_cfg import GoalCommandBaseCfg


@configclass
class LvlConsecutiveGoalCommandCfg(GoalCommandBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = LvlConsecutiveGoalCommand

    resample_distance_threshold: float = 0.2
    """Distance threshold for resampling the goals."""

    terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg()
    """Configuration for the trajectory sampling."""

    plot_points: bool = False
    """saves a plot of the level and type separated points"""
