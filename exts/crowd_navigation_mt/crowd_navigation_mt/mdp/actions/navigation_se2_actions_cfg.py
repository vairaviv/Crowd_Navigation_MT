# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .navigation_se2_actions import PerceptiveNavigationSE2Action


@configclass
class PerceptiveNavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PerceptiveNavigationSE2Action
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    use_raw_actions: bool = False
    """Whether to use raw actions or not."""
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""
    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    low_level_policy_file: str = MISSING
    """Path to the low level policy file."""
    observation_group: str = "policy"
    """Observation group to use for the low level policy."""
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    """Policy dependent scaling for the actions [vx, vy, w]."""
    reorder_joint_list: list[str] = MISSING
    """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
    trained with a different order."""


# @configclass
# class PerceptiveNavigationSE2ActionCfg(NavigationSE2ActionCfg):
#     class_type: type[ActionTerm] = PerceptiveNavigationSE2Action
#     """ Class of the action term."""
#     reorder_joint_list: list[str] = MISSING
#     """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
#     trained with a different order."""
