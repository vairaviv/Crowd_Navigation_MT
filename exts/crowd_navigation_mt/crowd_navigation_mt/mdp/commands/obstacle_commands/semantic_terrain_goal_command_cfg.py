# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from nav_collectors.terrain_analysis import TerrainAnalysisCfg

from .semantic_terrain_goal_command import SemanticGoalCommand
from .goal_command_base_cfg import GoalCommandBaseCfg


@configclass
class SemanticGoalCommandCfg(GoalCommandBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = SemanticGoalCommand

    plot_points: bool = False
    """saves a plot of the level and type separated points"""

    robot_radius: float = 0.5
    """robot radius for buffer calculations in order to have valid spawn and goal positions"""

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""

    sampling_radius: float = 5.0
    """max sampling radius for goal command"""

    