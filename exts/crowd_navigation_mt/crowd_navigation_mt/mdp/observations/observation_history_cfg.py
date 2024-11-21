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

if TYPE_CHECKING:
    from omni.isaac.lab.managers.action_manager import ActionTerm
    from omni.isaac.lab.managers.command_manager import CommandTerm
    from omni.isaac.lab.managers.manager_base import ManagerTermBase
    from omni.isaac.lab.managers.manager_term_cfg import ManagerTermBaseCfg

@configclass
class ObservationHistoryTermCfg(ObservationTermCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """

    history_length_actions: int = MISSING
    """How many actions should be stored"""
    
    history_length_positions: int = MISSING
    """How many positions should be stored"""

    history_time_span_actions: int = MISSING
    """The time span of the actions stored"""

    history_time_span_positions: int = MISSING
    """The time span of the positions stored"""


    # modifiers: list[ModifierCfg] | None = None
    # """The list of data modifiers to apply to the observation in order. Defaults to None,
    # in which case no modifications will be applied.

    # Modifiers are applied in the order they are specified in the list. They can be stateless
    # or stateful, and can be used to apply transformations to the observation data. For example,
    # a modifier can be used to normalize the observation data or to apply a rolling average.

    # For more information on modifiers, see the :class:`~omni.isaac.lab.utils.modifiers.ModifierCfg` class.
    # """

    # noise: NoiseCfg | None = None
    # """The noise to add to the observation. Defaults to None, in which case no noise is added."""

    # clip: tuple[float, float] | None = None
    # """The clipping range for the observation after adding noise. Defaults to None,
    # in which case no clipping is applied."""

    # scale: float | None = None
    # """The scale to apply to the observation after clipping. Defaults to None,
    # in which case no scaling is applied (same as setting scale to :obj:`1`)."""