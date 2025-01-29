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
class SemanticMapObsCfg(ObservationTermCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """

    obs_range: list[float] = [5.0, 5.0]
    """the observation size of the robot (x,y), in m"""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """The asset cfg from which the observation is made."""

    obstacle_cfg: SceneEntityCfg | None = None
    """given target cfg these targets will be represented in the semantic map as well."""

    target_buffer_radius: float = 0.2
    """the targets will be represented with a buffer in the semantic map, only used if target_cfg given."""

    obstacle_buffer_radius: float = 0.5
    """buffer radius for the dynamic obstacles in m"""

    plot_env_id: int = 0
    """if semantic map is plottes only one env will showed, defaults to env_0"""

    debug_plot: bool = False
    """if the observation map should be plotted"""
