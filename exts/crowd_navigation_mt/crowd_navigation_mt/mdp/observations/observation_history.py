from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.assets import AssetBase
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import ObservationTermCfg, ObservationManager

from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster

import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403
from .observation_history_cfg import ObservationHistoryTermCfg


from omni.isaac.lab.utils import math as math_utils
from ..actions import NavigationSE2Action
from .observations import base_position

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg


class ObservationHistory(ManagerTermBase):
    # def __init__(self, history_length_actions, history_length_positions: int = 1) -> None:
    #     """Initialize the buffers for the history of observations.

    #     Args:
    #         history_length: The length of the history.
    #     """

    #     self.buffers = {
    #         "actions": torch.empty(size=(0, history_length_actions, 3)),
    #         "positions": torch.empty(size=(0, history_length_positions, 2)),
    #     }

    def __init__(self, cfg: ObservationHistoryTermCfg, env: ManagerBasedEnv):
        """Initialize the observation history term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)

        self.cfg = cfg

        self.buffers = {
            "actions": torch.empty(size=(0, self.cfg.history_length_actions, 3)),
            "positions": torch.empty(size=(0, self.cfg.history_length_positions, 2)),
        }

    def reset(self, env: ManagerBasedRLEnv, buffer_names: list = None):
        """Reset the buffers for terminated episodes.

        Args:
            env: The environment object.
        """
        # Initialize & find terminated episodes
        try:
            terminated_mask = env.termination_manager.dones
        except AttributeError:
            terminated_mask = torch.zeros((env.num_envs), dtype=int).to(env.device)
        for key in buffer_names:
            # Initialize buffer if empty
            if self.buffers[key].shape[0] == 0:
                self.buffers[key] = torch.zeros((env.num_envs, *list(self.buffers[key].shape[1:]))).to(env.device)
            # Reset buffer for terminated episodes
            self.buffers[key][terminated_mask, :, :] = 0.0

    def get_history_of_actions(self, env: ManagerBasedRLEnv):
        """Get the history of actions.

        Args:
            env: The environment object.
        """
        # Reset buffer for terminated episodes
        self.reset(env, ["actions"])
        # Return updates buffer
        self.buffers["actions"] = self.buffers["actions"].roll(shifts=-1, dims=1)
        self.buffers["actions"][:, -1, :] = env.action_manager.action
        return self.buffers["actions"].reshape(env.num_envs, -1)

    def get_history_of_positions(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        """Get the history of positions.

        Args:
            env: The environment object.
            asset_cfg: The name of the asset.
        """
        # Reset buffer for terminated episodes
        self.reset(env, ["positions"])
        # Return updates buffer
        self.buffers["positions"] = self.buffers["positions"].roll(shifts=-1, dims=1)
        self.buffers["positions"][:, -1, :] = base_position(env, asset_cfg)[:, :2]
        return self.buffers["positions"].reshape(env.num_envs, -1)
    
"""
why should it be defined outside and is not recognized if its within the class
"""

# def get_history_of_positions(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
#         """Get the history of positions.

#         Args:
#             env: The environment object.
#             asset_cfg: The name of the asset.
#         """
#         # Reset buffer for terminated episodes
#         self.reset(env, ["positions"])
#         # Return updates buffer
#         self.buffers["positions"] = self.buffers["positions"].roll(shifts=-1, dims=1)
#         self.buffers["positions"][:, -1, :] = base_position(env, asset_cfg)[:, :2]
#         return self.buffers["positions"].reshape(env.num_envs, -1)
