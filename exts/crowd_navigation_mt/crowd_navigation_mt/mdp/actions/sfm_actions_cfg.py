# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os 

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .sfm_actions import SFMAction

@configclass
class SFMActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SFMAction
    """class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    use_raw_actions: bool = False
    """Whether to use raw actions or not."""
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""
    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""
    observation_group: str = "sfm_obstacle_control_obs"
    """Observation group to use for the low level policy."""
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    """Policy dependent scaling for the actions [vx, vy, w]."""
    robot_visible: bool = False
    """If the policy will take the agent into account"""
