from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import RigidObject

from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster

import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403

from omni.isaac.lab.utils import math as math_utils
from ..actions import NavigationSE2Action
from .observations import base_position

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg


class LidarHistory:
    def __init__(
        self,
        history_length: int = 1,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
        return_pose_history: bool = True,
        decimation: int = 1,
    ) -> None:
        """Initialize the buffers for the history of observations.

        History from old to new: [old, ..., new]

        Args:
            history_length: The length of the history.
            sensor_cfg: The sensor configuration.
            return_pose_history: Whether to return the history of poses.
            decimation: The decimation factor for the history.
        """
        self.sensor_cfg = sensor_cfg
        self.history_length = history_length * decimation
        self.lidar_buffer = None
        self.position_buffer = None
        self.yaw_buffer = None
        self.return_pose_history = return_pose_history
        self.decimation = decimation

    def reset(self, env: ManagerBasedRLEnv, sensor: RayCaster):
        """Reset the buffers for terminated episodes.

        Args:
            env: The environment object.
        """
        # Initialize & find terminated episodes
        try:
            terminated_mask = env.termination_manager.dones
        except AttributeError:
            terminated_mask = torch.ones((env.num_envs), dtype=int).to(env.device)
            # terminated_mask = torch.arange(0, env.num_envs, dtype=int).to(env.device)
        # Initialize buffer if empty
        if self.lidar_buffer is None or self.position_buffer is None or self.yaw_buffer is None:
            self.lidar_buffer = torch.zeros((env.num_envs, self.history_length, sensor.data.pos_w.shape[-1])).to(
                env.device
            )
            self.position_buffer = torch.zeros((env.num_envs, self.history_length + 1, 3)).to(env.device)
            self.yaw_buffer = torch.zeros((env.num_envs, self.history_length + 1)).to(env.device)
        # Reset buffer for terminated episodes
        self.lidar_buffer[terminated_mask, :, :] = 0.0
        self.position_buffer[terminated_mask, :, :] = 0.0
        self.yaw_buffer[terminated_mask, :] = 0.0

        # return torch.nonzero(terminated_mask).flatten()
        return terminated_mask

    def get_history(self, env: ManagerBasedRLEnv):
        """Get the history of actions.

        Args:
            env: The environment object.
        """
        sensor: RayCaster = env.scene.sensors[self.sensor_cfg.name]
        # Reset buffer for terminated episodes
        reset_idx = self.reset(env, sensor)

        # update buffers
        # Return updates buffer
        self.lidar_buffer = self.lidar_buffer.roll(shifts=-1, dims=1)
        self.position_buffer = self.position_buffer.roll(shifts=-11, dims=1)
        self.yaw_buffer = self.yaw_buffer.roll(shifts=-11, dims=1)

        self.yaw_buffer[:, -1] = math_utils.axis_angle_from_quat(math_utils.yaw_quat(sensor.data.quat_w))[
            :, 2
        ]  # sensor yaw
        distances = sensor.data.distances
        # distances = shift_point_indices_by_heading(distances, self.yaw_buffer[:, 0])
        distances[torch.isinf(distances)] = 0.0
        self.lidar_buffer[:, -1, :] = distances  # lidar distances
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