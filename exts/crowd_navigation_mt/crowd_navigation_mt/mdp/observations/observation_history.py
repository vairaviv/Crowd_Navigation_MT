from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

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

    def __init__(self, cfg: ObservationHistoryTermCfg, env: ManagerBasedRLEnv):
        """Initialize the observation history term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)

        self._cfg = cfg
        self._env = env

        # TODO updated the buffers only if the time is up
        # self.update_time_actions = self._cfg.history_time_span_actions/self._cfg.history_length_actions
        # self.update_time_positions = self._cfg.history_time_span_positions/self._cfg.history_length_positions
        
        self.buffers = {
            "actions": torch.empty(size=(0, self._cfg.history_length_actions, 3)),
            "positions": torch.empty(size=(0, self._cfg.history_length_positions, 2)),
        }

    def __call__(self, *args, method) -> torch.Any:

        method = getattr(self, method, None)

        if method is None or not callable(method):
            raise ValueError(f"Method '{method}' not found in the class.")

        # Reset buffer for terminated episodes
        try:
            env_ids = self._env.termination_manager.dones
        except AttributeError:
            env_ids = torch.arange(self.num_envs)

        self.reset(env_ids)

        return method(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None, buffer_names: list = ["actions", "positions"], *args):
        """Reset the buffers for terminated episodes.

        Args:
            env: The environment object.
        """
        if env_ids is None:
            return
        
        terminated_mask = torch.zeros((self.num_envs), dtype=torch.bool)
        terminated_mask[env_ids] = True

        for key in buffer_names:
            if self.buffers[key].shape[0] == 0:
                self.buffers[key] = torch.zeros((self.num_envs, *list(self.buffers[key].shape[1:]))).to(device=self.device)

            self.buffers[key][terminated_mask, :, :] = 0.0
    
    def get_history_of_actions(self, env_ids: Sequence[int] | None = None):
        """Get the history of actions.

        Args:
            env: The environment object.
        """
        
        self.buffers["actions"] = self.buffers["actions"].roll(shifts=-1, dims=1)
        self.buffers["actions"][:, -1, :] = self._env.action_manager.action
        return self.buffers["actions"].reshape(self._env.num_envs, -1)

    def get_history_of_positions(self, env_ids: Sequence[int] | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        """Get the history of positions.

        Args:
            env: The environment object.
            asset_cfg: The name of the asset.
        """

        self.buffers["positions"] = self.buffers["positions"].roll(shifts=-1, dims=1)
        self.buffers["positions"][:, -1, :] = base_position(self._env, asset_cfg)[:, :2]
        return self.buffers["positions"].reshape(self._env.num_envs, -1)
