# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.modifiers import ModifierCfg
from omni.isaac.lab.utils.noise import NoiseCfg
from omni.isaac.lab.managers.manager_term_cfg import ManagerTermBaseCfg, ObservationTermCfg
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.managers.action_manager import ActionTerm
    from omni.isaac.lab.managers.command_manager import CommandTerm
    from omni.isaac.lab.managers.manager_base import ManagerTermBase
    from omni.isaac.lab.managers.manager_term_cfg import ManagerTermBaseCfg


@configclass
class LidarHistoryTermCfg(ObservationTermCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """

    history_length: int = 1
    "num entries of the history"

    history_time_span: int = 1
    "time span for the history of the lidar distances"
    
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")
    
    return_pose_history: bool = True
    
    decimation: int = 1

    