# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from nav_collectors.collectors import TrajectorySamplingCfg

from .obst_goal_command import ObstGoalCommand
from .goal_command_base_cfg import GoalCommandBaseCfg


@configclass
class ObstGoalCommandCfg(GoalCommandBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = ObstGoalCommand

    trajectory_config: dict = {
        "num_paths": [100],
        "max_path_length": [10.0],
        "min_path_length": [2.0],
    }
    """Configuration for the trajectory. Contains list of different trajectory configurations."""

    z_offset_spawn: float = 0.1
    """Offset in z direction for the spawn height."""

    infite_sampling: bool = True
    """Enable sampling of the same start-goal pairs multiple times."""

    max_trajectories: int | None = None
    """Maximum number of trajectories to sample."""

    traj_sampling: TrajectorySamplingCfg = TrajectorySamplingCfg()
    """Configuration for the trajectory sampling."""

    reset_pos_term_name: str | None = "reset_base"
    """Name of the termination term that resets the base position.

    This term is normally called before the goal resample and therefore with the old commands. To fix this, we
    call it again after the goal resample."""
