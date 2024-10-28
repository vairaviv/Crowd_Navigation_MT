# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os 

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .navigation_se2_actions import PerceptiveNavigationSE2Action
from crowd_navigation_mt import CROWDNAV_DATA_DIR

# TODO has to be a way to solve it cleaner

# CROWD_NAV_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
# CROWD_NAV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
# NAVSUITE_EXT_DIR = os.path.join(CROWD_NAV_DIR, "isaac-nav-suite", "exts")
# NAVSUITE_TASKS_DATA_DIR = os.path.join(NAVSUITE_EXT_DIR, "nav_tasks", "data")

ISAAC_GYM_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]

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
    low_level_policy_file: str = os.path.join(CROWDNAV_DATA_DIR, "Policies", "perceptive_locomotion_jit.pt")
    """Path to the low level policy file."""
    observation_group: str = "policy"
    """Observation group to use for the low level policy."""
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    """Policy dependent scaling for the actions [vx, vy, w]."""
    reorder_joint_list: list[str] = ISAAC_GYM_JOINT_NAMES
    """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
    trained with a different order."""


# @configclass
# class PerceptiveNavigationSE2ActionCfg(NavigationSE2ActionCfg):
#     class_type: type[ActionTerm] = PerceptiveNavigationSE2Action
#     """ Class of the action term."""
#     reorder_joint_list: list[str] = MISSING
#     """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
#     trained with a different order."""
