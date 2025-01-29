# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from nav_collectors.terrain_analysis import TerrainAnalysisCfg

from .semantic_terrain_consecutive_goal_command import SemanticConsecutiveGoalCommand
from .semantic_terrain_goal_command_cfg import SemanticGoalCommandCfg


@configclass
class SemanticConsecutiveGoalCommandCfg(SemanticGoalCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = SemanticConsecutiveGoalCommand

    num_sfm_obstacle: int = 10
    """amount of dynamic obstacles operated by the action, the rest is stored under the terrain"""

    resample_distance_threshold: float = 0.2
    """"if the threshold is achieve, then automatically the next goal is commanded within sampling_radius"""