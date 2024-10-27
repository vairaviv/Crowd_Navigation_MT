from __future__ import annotations

from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, AssetBase
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg,ManagerBasedRLEnv , ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass, math as math_utils
from omni.isaac.lab.utils.assets import check_file_path, read_file


from dataclasses import MISSING


import torch
import math
import crowd_navigation_mt.mdp as mdp

from crowd_navigation_mt.mdp.commands.terrain_analysis import (
    TerrainAnalysis,
    TerrainAnalysisCfg,
)

#from icecream import ic


"""Obstacle Position controller action"""
from omni.isaac.lab.utils.warp import raycast_dynamic_meshes
from omni.isaac.lab.sensors import RayCaster, patterns, RayCasterCfg
import numpy as np
import warp as wp


class ObstacleActionTermSimple(ActionTerm):
    """Simple action term that moves around an obstacle."""

    _asset: RigidObject
    cfg: ObstacleActionTermSimpleCfg
    _env: ManagerBasedRLEnv
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: ObstacleActionTermSimpleCfg, env: ManagerBasedRLEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        # self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        # self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        self.max_velocity = cfg.max_velocity
        self.max_acceleration = cfg.max_acceleration
        self.max_rotvel = cfg.max_rotvel
        self.dt = env.step_dt
        self.env_spacing = env.scene.cfg.env_spacing
        self.env = env
        self.name = cfg.asset_name

        self.arena_size = math.sqrt(env.num_envs) * self.env_spacing + self.env_spacing

        self.p_gain = 5
        self.d_gain = 0
        self.p_gain_rot = 5

        # raycaster to keep the obstacles on the ground
        if isinstance(self._env.scene.sensors[self.cfg.raycaster_sensor], RayCaster):
            self._raycaster: RayCaster = self._env.scene.sensors[self.cfg.raycaster_sensor]
        else:
            raise ValueError(f"Sensor {self.cfg.raycaster_sensor} is not a RayCaster sensor")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # should not learn this action so we say the action shape is empty
        return 0  # self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty()  # self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._vel_command

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):

        pass

    def apply_actions(self):

        ##
        # 2D position control
        ##

        if self.max_velocity == 0:
            self._asset.write_root_velocity_to_sim(torch.zeros_like(self._vel_command))
            return

        # position error
        target_positions = self._env.observation_manager.compute_group(group_name="obstacle_control")
        # target_positions = command_2d_pos[:, :2]

        current_positions = self._asset.data.root_pos_w[:, :2]

        pos_error = target_positions - current_positions

        # des velocity
        des_vel = pos_error
        scale_factors = torch.clamp(self.max_velocity / torch.norm(des_vel, dim=1), max=1.0)
        des_vel *= scale_factors.unsqueeze(1)

        # velocity error
        current_velocity = self._asset.data.root_lin_vel_w[:, :2]

        velocity_error = des_vel - current_velocity

        # pd velocity controller
        acceleration = (current_velocity - self._vel_command[:, :2]) / self.dt

        acceleration_command = self.p_gain * velocity_error - self.d_gain * acceleration
        scale_factors = torch.clamp(self.max_acceleration / torch.norm(acceleration_command, dim=1), max=1.0)
        acceleration_command *= scale_factors.unsqueeze(1)

        vel_command = current_velocity + acceleration_command * self.dt

        # force the orientation to be upright
        # self._asset.data.root_state_w[:, 3:7] = self._orientations

        ##
        # orientation control
        ##
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(self._asset.data.body_quat_w.squeeze())
        des_yaw = torch.atan2(pos_error[:, 1], pos_error[:, 0])
        des_roll, des_pitch = torch.zeros_like(roll), torch.zeros_like(pitch)

        des_quat = math_utils.quat_from_euler_xyz(des_roll, des_pitch, des_yaw)
        quat_error = math_utils.quat_mul(des_quat, math_utils.quat_inv(self._asset.data.body_quat_w.squeeze()))
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error) * self.p_gain_rot

        scale_factors = torch.clamp(self.max_rotvel / torch.norm(axis_angle_error, dim=1), max=1.0)
        des_rotvel = axis_angle_error * scale_factors.unsqueeze(1)

        ##
        # set velocity targets
        ##

        self._vel_command[:, :2] = vel_command
        self._vel_command[:, 3] = torch.zeros(self.num_envs)
        self._vel_command[:, 3:] = des_rotvel

        self._asset.write_root_velocity_to_sim(self._vel_command)

        ##
        # set z position
        ##
        ray_casts = self._raycaster.data.ray_hits_w[..., 2]
        ray_casts[torch.isinf(ray_casts)] = -1000
        z_heights = torch.max(ray_casts, dim=1).values + self.cfg.obstacle_center_height

        if self._env.common_step_counter > 0:
            self._asset.data.root_pos_w[:, 2] = z_heights

            self._asset.write_root_pose_to_sim(self._asset.data.root_state_w[:, :7])

        else:
            # set the initial position to grid origins
            self._asset.data.root_pos_w[:, :2] = self.env.scene.env_origins[:, :2] + torch.tensor(
                self.env.scene.rigid_objects[self.name].cfg.init_state.pos
            )[:2].to(self.device).unsqueeze(0)
            self._asset.data.root_pos_w[:, 2] = z_heights

            self._asset.write_root_pose_to_sim(self._asset.data.root_state_w[:, :7])


@configclass
class ObstacleActionTermSimpleCfg(ActionTermCfg):
    """Configuration for the action term."""

    class_type: type = ObstacleActionTermSimple
    """The class corresponding to the action term."""

    max_velocity: float = 5.0

    max_acceleration: float = 10.0

    max_rotvel: float = 1.0

    # terrain_analysis = MISSING
    raycaster_sensor: str = MISSING

    obstacle_center_height: float = 1.0

    # terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg()
    """Terrain analysis configuration."""


"""Velocity controller action for Anymal robot"""


class NavigationSE2Action(ActionTerm):
    """Actions to navigate a robot by following some path."""

    cfg: NavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: NavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # check if policy file exists
        if not check_file_path(cfg.low_level_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.low_level_policy_file}' does not exist.")
        # load policies
        file_bytes = read_file(self.cfg.low_level_policy_file)
        self.low_level_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.low_level_policy = torch.jit.freeze(self.low_level_policy.eval())

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = 3  # [vx, vy, omega]

        # set up buffers
        self._init_buffers()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    @property
    def prev_low_level_actions(self) -> torch.Tensor:
        return self._prev_low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process low-level navigation actions. This function is called with a frequency of 10Hz"""

        # Store low level navigation actions
        self._raw_navigation_velocity_actions[:] = actions
        # scale actions:
        self._processed_navigation_velocity_actions = self._raw_navigation_velocity_actions * self._scale + self._offset
        # reshape into 3D path
        # self._processed_navigation_velocity_actions[:] = actions.clone().view(self.num_envs, self._action_dim)

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_actions[:] = self._low_level_actions.clone()
            # Get low level actions from low level policy
            self._low_level_actions[:] = self.low_level_policy(
                self._env.observation_manager.compute_group(group_name="lowlevel_policy")
            )
            # Process low level actions
            self.low_level_action_term.process_actions(self._low_level_actions)

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_level_actions = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._prev_low_level_actions = torch.zeros_like(self._low_level_actions)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt
        self._counter = 0


@configclass
class NavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavigationSE2Action
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )
    """Configuration of the low level action term."""
    low_level_policy_file: str = MISSING
    """Path to the low level policy file. Has to be a torchscript file."""
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""
    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""
