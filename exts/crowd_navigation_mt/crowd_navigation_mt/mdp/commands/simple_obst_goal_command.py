# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from .commands_cfg import SimpleObstGoalCommandCfg


class SimpleObstGoalCommand(CommandTerm):
    """Command generator that does nothing.

    This command generator does not generate any commands. It is used for environments that do not
    require any commands.
    """

    cfg: SimpleObstGoalCommandCfg
    """Configuration for the command generator."""

    def __str__(self) -> str:
        msg = "SimpleObstacleGoalCommand:\n"
        msg += f"\tCommand dimension: {self.cfg.resampling_time_range}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self):
        """Simple static command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float):
        pass

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass