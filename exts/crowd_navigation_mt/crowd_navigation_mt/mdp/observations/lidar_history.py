from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import ManagerTermBase
from omni.isaac.lab.sensors import RayCaster

import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403
from .lidar_history_cfg import LidarHistoryTermCfg

from omni.isaac.lab.utils import math as math_utils


class LidarHistory(ManagerTermBase):
    # def __init__(
    #     self,
    #     history_length: int = 1,
    #     sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    #     return_pose_history: bool = True,
    #     decimation: int = 1,
    # ) -> None:
    #     """Initialize the buffers for the history of observations.

    #     History from old to new: [old, ..., new]

    #     Args:
    #         history_length: The length of the history.
    #         sensor_cfg: The sensor configuration.
    #         return_pose_history: Whether to return the history of poses.
    #         decimation: The decimation factor for the history.
    #     """
    #     self.sensor_cfg = sensor_cfg
    #     self.history_length = history_length * decimation
    #     self.lidar_buffer = None
    #     self.position_buffer = None
    #     self.yaw_buffer = None
    #     self.return_pose_history = return_pose_history
    #     self.decimation = decimation
    
    def __init__(self, cfg: LidarHistoryTermCfg, env: ManagerBasedRLEnv):
        """Initialize the lidar history term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """

        super().__init__(cfg, env)

        self._cfg = cfg
        self._env = env
        
        self.lidar_buffer = None
        self.position_buffer = None
        self.yaw_buffer = None

        # TODO check wheter it is necessary to have them here or if they can me directly accessed through self.cfg....
        self.history_length = self._cfg.history_length * self._cfg.decimation
        self.return_pose_history = self._cfg.return_pose_history
        self.decimation = self._cfg.decimation
        self.sensor_cfg = self._cfg.sensor_cfg

    def __call__(self, *args, method) -> torch.Any:

        # method_name = method.get("method")

        method = getattr(self, method, None)

        if method is None or not callable(method):
            raise ValueError(f"Method '{method_name}' not found in the class.")
            
        return method(*args)

    def reset(self, env_ids: Sequence[int] | None = None): # , env: ManagerBasedRLEnv, sensor: RayCaster
        """Resets the manager term and the buffers created for lidar history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # Initialize buffer if empty
        if self.lidar_buffer is None or self.position_buffer is None or self.yaw_buffer is None:
            # self.lidar_buffer = torch.zeros((env.num_envs, self.history_length, sensor.data.pos_w.shape[-1])).to(
            #     env.device
            # )
            lidar = self._env.scene.sensors[self.sensor_cfg.name]
            self.lidar_buffer = torch.zeros((self.num_envs, self.history_length, lidar.data.distances.shape[-1])).to(
                self.device
            )
            self.position_buffer = torch.zeros((self.num_envs, self.history_length + 1, 3)).to(self.device)
            self.yaw_buffer = torch.zeros((self.num_envs, self.history_length + 1)).to(self.device)
        
        if env_ids is None:
            try:
                env_ids = torch.nonzero(self._env.termination_manager.dones).flatten()
            except AttributeError:
                # env_ids = torch.ones((self.num_envs), dtype=torch.bool).to(self.device)
                env_ids = torch.arange(0, self.num_envs, dtype=int).to(self.device)

        # Reset buffer for terminated episodes
        self.lidar_buffer[env_ids, :, :] = 0.0
        self.position_buffer[env_ids, :, :] = 0.0
        self.yaw_buffer[env_ids, :] = 0.0

        # return torch.nonzero(terminated_mask).flatten()

        # # Initialize & find terminated episodes
        # try:
        #     terminated_mask = env.termination_manager.dones
        # except AttributeError:
        #     terminated_mask = torch.ones((env.num_envs), dtype=int).to(env.device)
        #     # terminated_mask = torch.arange(0, env.num_envs, dtype=int).to(env.device)
        # # Initialize buffer if empty
        # if self.lidar_buffer is None or self.position_buffer is None or self.yaw_buffer is None:
        #     # self.lidar_buffer = torch.zeros((env.num_envs, self.history_length, sensor.data.pos_w.shape[-1])).to(
        #     #     env.device
        #     # )
        #     self.lidar_buffer = torch.zeros((env.num_envs, self.history_length, sensor.data.distances.shape[-1])).to(
        #         env.device
        #     )
        #     self.position_buffer = torch.zeros((env.num_envs, self.history_length + 1, 3)).to(env.device)
        #     self.yaw_buffer = torch.zeros((env.num_envs, self.history_length + 1)).to(env.device)
        # # Reset buffer for terminated episodes
        # self.lidar_buffer[terminated_mask, :, :] = 0.0
        # self.position_buffer[terminated_mask, :, :] = 0.0
        # self.yaw_buffer[terminated_mask, :] = 0.0

        # # return torch.nonzero(terminated_mask).flatten()
        # return torch.nonzero(env_ids).flatten()
        return env_ids

    def get_history(self, env: ManagerBasedRLEnv):
        """Get the history of actions.

        Args:
            env: The environment object.
        """
        sensor: RayCaster = env.scene.sensors[self.sensor_cfg.name]
        # Reset buffer for terminated episodes
        reset_idx = self.reset()

        # update buffers
        # Return updates buffer
        self.lidar_buffer = self.lidar_buffer.roll(shifts=-1, dims=1)
        self.position_buffer = self.position_buffer.roll(shifts=-11, dims=1)
        self.yaw_buffer = self.yaw_buffer.roll(shifts=-11, dims=1)

        self.yaw_buffer[:, -1] = math_utils.axis_angle_from_quat(math_utils.yaw_quat(sensor.data.quat_w))[
            :, 2
        ]  # sensor yaw

        # distances = torch.linalg.vector_norm(sensor.data.ray_hits_w, dim=-1)
        distances = sensor.data.distances
        # distances = shift_point_indices_by_heading(distances, self.yaw_buffer[:, 0])
        # before all the infinite sensor measurements were set to 0.0, why?
        distances[torch.isinf(distances)] = 0.0
        # distances[torch.isinf(distances)] = sensor.cfg.max_distance

        self.lidar_buffer[:, -1, :] = distances  # lidar distances

        # # TODO this needs to happen in the Raycaster and not her in the history
        # sensor._data.distances = distances

        # # this is to handle the case where distances has a different shape, as only a few environments are reset,
        # # due to temination, maybe there is a way to handle this better check TODO
        # self.lidar_buffer_cloned = self.lidar_buffer.clone().detach()
        # if reset_idx is not None and len(reset_idx) > 0:
        #     self.lidar_buffer_cloned[reset_idx, -1, :] = distances
        # else:
        #     self.lidar_buffer_cloned[:, -1, :] = distances

        # self.lidar_buffer = self.lidar_buffer_cloned

        self.position_buffer[:, -1, :] = sensor.data.pos_w  # sensor positions world frame

        # reset relative positions and yaw for terminated episodes
        if reset_idx.any():
            self.position_buffer[reset_idx] = (
                sensor.data.pos_w[reset_idx].unsqueeze(1).repeat(1, self.history_length + 1, 1)
            )
            self.yaw_buffer[reset_idx] = self.yaw_buffer[reset_idx, 0].unsqueeze(1).repeat(1, self.history_length + 1)

        # update relative positions and yaw
        relative_history_positions = self.position_buffer[..., :2] - self.position_buffer[:, :1, :2]
        relative_history_yaw = math_utils.wrap_to_pi(self.yaw_buffer - self.yaw_buffer[:, :1])

        history_pose = torch.cat((relative_history_positions, relative_history_yaw.unsqueeze(-1)), dim=-1)

        # reset if jump in position
        if self.history_length > 1:
            jumped = torch.norm(relative_history_positions[:, 1], dim=1) > 1.5
            if jumped.any():
                self.position_buffer[jumped] = (
                    sensor.data.pos_w[jumped].unsqueeze(1).repeat(1, self.history_length + 1, 1)
                )
                self.yaw_buffer[jumped] = self.yaw_buffer[jumped, 0].unsqueeze(1).repeat(1, self.history_length + 1)
                self.lidar_buffer[jumped, 1:, :] = 0.0
                history_pose[jumped] = 0.0

        # adds additional 3 observations, which is the difference in position of the robot from the previous to the 
        # current step
        if self.return_pose_history:
            lidar_buffer_flattened = self.lidar_buffer[:, :: self.decimation, :].reshape(self.lidar_buffer.size(0), -1)
            history_pose_flattened = history_pose[:, 1 :: self.decimation, :].reshape(history_pose.size(0), -1)
            full_history = torch.cat((lidar_buffer_flattened, history_pose_flattened), dim=1)
        else:
            full_history = self.lidar_buffer[:, :: self.decimation, :]

        # # # for debugging
        # import matplotlib.pyplot as plt
        # import numpy as np
        # index = 33
        # lidar_history_d = self.lidar_buffer[:, :: self.decimation, :]
        # colors = plt.cm.viridis(np.linspace(0, 1, int(self.history_length/self.decimation)))  # Using the 'viridis' colormap
        # for i in range(int(self.history_length/self.decimation)):
        #     distances_ = lidar_history_d[:, i, :].detach()
        #     n_points = len(distances_[0])
        #     distances_[distances_ > 10] = 0.0
        #     distances_ = distances_[index].cpu()
        #     degs = torch.linspace(0, 2 * 3.14159265358979, n_points)
        #     p_x = distances_ * torch.cos(degs)
        #     p_y = distances_ * torch.sin(degs)
        #     plt.plot(p_x, p_y, "o", label=f"t={i}", color=colors[i])
        # plt.axis("equal")
        # plt.legend()
        # plt.show()

        # select the decimated history

        return full_history



# def get_history(self, env: ManagerBasedRLEnv):
#     """Get the history of actions.

#     Args:
#         env: The environment object.
#     """
#     sensor: RayCaster = env.scene.sensors[self.sensor_cfg.name]
#     # Reset buffer for terminated episodes
#     reset_idx = self.reset(env, sensor)

#     # update buffers
#     # Return updates buffer
#     self.lidar_buffer = self.lidar_buffer.roll(shifts=-1, dims=1)
#     self.position_buffer = self.position_buffer.roll(shifts=-11, dims=1)
#     self.yaw_buffer = self.yaw_buffer.roll(shifts=-11, dims=1)

#     self.yaw_buffer[:, -1] = math_utils.axis_angle_from_quat(math_utils.yaw_quat(sensor.data.quat_w))[
#         :, 2
#     ]  # sensor yaw
#     distances = sensor.data.distances
#     # distances = shift_point_indices_by_heading(distances, self.yaw_buffer[:, 0])
#     distances[torch.isinf(distances)] = 0.0
#     self.lidar_buffer[:, -1, :] = distances  # lidar distances
#     self.position_buffer[:, -1, :] = sensor.data.pos_w  # sensor positions world frame

#     # reset relative positions and yaw for terminated episodes
#     if reset_idx.any():
#         self.position_buffer[reset_idx] = (
#             sensor.data.pos_w[reset_idx].unsqueeze(1).repeat(1, self.history_length + 1, 1)
#         )
#         self.yaw_buffer[reset_idx] = self.yaw_buffer[reset_idx, 0].unsqueeze(1).repeat(1, self.history_length + 1)

#     # update relative positions and yaw
#     relative_history_positions = self.position_buffer[..., :2] - self.position_buffer[:, :1, :2]
#     relative_history_yaw = math_utils.wrap_to_pi(self.yaw_buffer - self.yaw_buffer[:, :1])

#     history_pose = torch.cat((relative_history_positions, relative_history_yaw.unsqueeze(-1)), dim=-1)

#     # reset if jump in position
#     if self.history_length > 1:
#         jumped = torch.norm(relative_history_positions[:, 1], dim=1) > 1.5
#         if jumped.any():
#             self.position_buffer[jumped] = (
#                 sensor.data.pos_w[jumped].unsqueeze(1).repeat(1, self.history_length + 1, 1)
#             )
#             self.yaw_buffer[jumped] = self.yaw_buffer[jumped, 0].unsqueeze(1).repeat(1, self.history_length + 1)
#             self.lidar_buffer[jumped, 1:, :] = 0.0
#             history_pose[jumped] = 0.0

#     if self.return_pose_history:
#         lidar_buffer_flattened = self.lidar_buffer[:, :: self.decimation, :].reshape(self.lidar_buffer.size(0), -1)
#         history_pose_flattened = history_pose[:, 1 :: self.decimation, :].reshape(history_pose.size(0), -1)
#         full_history = torch.cat((lidar_buffer_flattened, history_pose_flattened), dim=1)
#     else:
#         full_history = self.lidar_buffer[:, :: self.decimation, :]

#     # # # for debugging
#     # import matplotlib.pyplot as plt
#     # import numpy as np
#     # index = 33
#     # lidar_history_d = self.lidar_buffer[:, :: self.decimation, :]
#     # colors = plt.cm.viridis(np.linspace(0, 1, int(self.history_length/self.decimation)))  # Using the 'viridis' colormap
#     # for i in range(int(self.history_length/self.decimation)):
#     #     distances_ = lidar_history_d[:, i, :].detach()
#     #     n_points = len(distances_[0])
#     #     distances_[distances_ > 10] = 0.0
#     #     distances_ = distances_[index].cpu()
#     #     degs = torch.linspace(0, 2 * 3.14159265358979, n_points)
#     #     p_x = distances_ * torch.cos(degs)
#     #     p_y = distances_ * torch.sin(degs)
#     #     plt.plot(p_x, p_y, "o", label=f"t={i}", color=colors[i])
#     # plt.axis("equal")
#     # plt.legend()
#     # plt.show()

#     # select the decimated history

#     return full_history