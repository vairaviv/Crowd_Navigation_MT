from __future__ import annotations


from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.managers import ActionTerm, ActionManager, ActionTermCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.utils import math as math_utils


from typing import TYPE_CHECKING
from dataclasses import MISSING

import torch
import math
import crowd_navigation_mt.mdp as mdp


if TYPE_CHECKING:
    from .actions_cfg import SimpleDynObstacleActionTermCfg



class SimpleDynObstacleActionTerm(ActionTerm):
    """Simple dynamic Obstacle action term, PD control action"""

    _asset: RigidObject
    cfg: SimpleDynObstacleActionTermCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SimpleDynObstacleActionTermCfg, env: ManagerBasedRLEnv):

        # call super constructor    
        super().__init__(cfg, env)

        # create buffers
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        self.max_velocity = cfg.max_velocity
        self.max_acceleration = cfg.max_acceleration
        self.max_rotvel = cfg.max_rotvel
        self.dt = env.step_dt
        self.env_spacing = env.scene.cfg.env_spacing
        self.env = env
        self.name = cfg.asset_name

        self.arena_size = math.sqrt(env.num_envs) * self.env_spacing + self.env_spacing

        self.p_gain = 10
        self.d_gain = 2
        self.p_gain_rot = 10

        # # raycaster to keep the obstacles on the ground
        # # currently no raycaster implemented for the simple obstacle
        # 
        # if isinstance(self._env.scene.sensors[self.cfg.raycaster_sensor], RayCaster):
        #     self._raycaster: RayCaster = self._env.scene.sensors[self.cfg.raycaster_sensor]
        # else:
        #     raise ValueError(f"Sensor {self.cfg.raycaster_sensor} is not a RayCaster sensor")



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
            # behaves like a static obstacle 
            self._asset.write_root_velocity_to_sim(torch.zeros_like(self._vel_command))
            
            return
        
        # position error
        # understand how the compute_group works with group name and where it is implemented
        # it is implemented as the observation cfg for the obstacles defined in env_cfg files
        target_positions = self._env.observation_manager.compute_group(group_name="dyn_obstacle_control")[:,:2]
        current_positions = self._asset.data.root_pos_w[:, :2]

        pos_error = target_positions - current_positions

        # desired velocity
        des_vel = pos_error
        scale_factors = torch.clamp(self.max_velocity / torch.norm(des_vel, dim=1), max=1.0)
        
        des_vel *= scale_factors.unsqueeze(1)

        # velocity error
        current_velocity = self._asset.data.root_lin_vel_w[:,:2]
        velocity_error = des_vel - current_velocity

        # pd velocity controller
        acceleration = (current_velocity - self._vel_command[:,:2]) / self.dt

        acceleration_command = self.p_gain * velocity_error - self.d_gain * acceleration 
        scale_factors = torch.clamp(self.max_acceleration / torch.norm(acceleration_command, dim=1), max=1.0)
        acceleration_command *= scale_factors.unsqueeze(1)

        velocity_command = current_velocity + acceleration_command * self.dt


        ##
        # orientation control
        ##

        # object orientation
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(self._asset.data.body_quat_w.squeeze())
        
        # desired object orientation
        des_yaw = torch.atan2(pos_error[:,1], pos_error[:,0])
        des_roll, des_pitch = torch.zeros_like(roll), torch.zeros_like(pitch)
        des_quat = math_utils.quat_from_euler_xyz(des_roll, des_pitch, des_yaw)
        
        # orientation error
        quat_error = math_utils.quat_mul(des_quat, math_utils.quat_inv(self._asset.data.body_quat_w.squeeze()))
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)

        # p controller for yaw axis 
        axis_angle_command = axis_angle_error * self.p_gain_rot
        scale_factors = torch.clamp(self.max_rotvel / torch.norm(axis_angle_command, dim=1), max=1.0)
        rotation_command = axis_angle_command * scale_factors.unsqueeze(1)


        ##
        # set velocity
        ##

        self._vel_command[:,:2] = velocity_command
        self._vel_command[:,3] = torch.zeros(self.num_envs)
        self._vel_command[:,3:] = rotation_command

        # pass new velocity to sim
        self._asset.write_root_velocity_to_sim(self._vel_command)

        
        # first implementation without raycaster
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
