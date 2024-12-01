# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .goal_command_base import GoalCommandBaseTerm


@configclass
class GoalCommandBaseCfg(CommandTermCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = GoalCommandBaseTerm

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
