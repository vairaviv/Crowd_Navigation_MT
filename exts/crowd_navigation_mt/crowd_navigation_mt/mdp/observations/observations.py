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

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg


def expanded_generated_commands(env: ManagerBasedRLEnvCfg, command_name: str, size: int) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name, with a dimension added such
    that the total size is equal to the given size.
    Useful for expanding the generated command to similar shape to the other observations.
    """
    command = env.command_manager.get_command(command_name)
    dim_expansion = size // command[0].numel()  # Expand each eviroment's command to the given size
    return command.unsqueeze(2).expand(-1, -1, dim_expansion).reshape(command.shape[0], -1)


##################################################
# PLR Observations
##################################################

"""helper"""


def transform_w_points_to_b_points(
    points_w: torch.tensor, w_r_wb: torch.tensor, quat, yaw: bool = False
) -> torch.Tensor:
    """Transform points from the world frame to the sensor's frame,
    given the sensor's position and orientation in the world frame."""

    # TODO check if correct and if it can be simplified
    # shift points to the sensor's frame origin
    points_w_shifted = math_utils.transform_points(points=points_w, pos=-w_r_wb)
    # convert inf to nan values
    pc_w_shifted = torch.where(torch.isinf(points_w_shifted), torch.tensor(float("NaN")), points_w_shifted)

    # rotate the points to the sensor's frame
    quat_inv = math_utils.quat_inv(quat) if not yaw else math_utils.yaw_quat(math_utils.quat_inv(quat))
    points_b = math_utils.transform_points(points=pc_w_shifted, quat=quat_inv)

    return points_b


# def shift_point_indices_by_heading(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
#     """Shift indices of points by the heading angle."""
#     num_points = points.shape[1]
#     angle_resolution = 2 * torch.pi / num_points

#     # Calculate shifted indices
#     shifted_indices = torch.round(yaw / angle_resolution).long() % num_points

#     # Create a tensor of indices
#     original_indices = torch.arange(num_points, device=points.device).unsqueeze(0).repeat(points.shape[0], 1)

#     # Calculate the new indices for all points, adjusting for the negative shift
#     new_indices = (original_indices + shifted_indices.unsqueeze(1)) % num_points

#     # Use gather to shift the points
#     points_shifted = torch.gather(points, 1, new_indices)

#     return points_shifted


# # memory efficient version, but uses loop
# def shift_point_indices_by_heading(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
#     """Shift indices of points by the heading angle."""
#     num_points = points.shape[1]
#     angle_resolution = 2 * torch.pi / num_points

#     # Calculate shifted indices
#     shifted_indices = torch.round(yaw / angle_resolution).long() % num_points

#     # Initialize an empty tensor for the shifted points
#     points_shifted = torch.empty_like(points)

#     # Efficiently shift the points
#     for i in range(points.shape[0]):
#         points_shifted[i] = torch.roll(points[i], -shifted_indices[i].item(), dims=0)

#     return points_shifted


"""Sensors"""


def heigh_scan_binary(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = sensor.data.distances
    mean_dist = distances.mean(dim=1)
    distances = distances <= mean_dist.unsqueeze(1)

    return distances.float()


# import matplotlib.pyplot as plt
# plt.imshow(distances.view(env.num_envs, 41, 41)[0].cpu())
# plt.show()


def lidar_obs(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, yaw_only: bool = False) -> torch.Tensor:
    """lidar scan from the given sensor. returns a pointcloud in the sensor's frame."""
    # TODO: check if this works
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pc_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w
    sensor_quat_w = sensor.data.quat_w

    ray_hits_b = transform_w_points_to_b_points(pc_w, sensor_pos_w, sensor_quat_w, yaw=yaw_only)
    ray_hits_b = torch.where(torch.isnan(ray_hits_b), torch.tensor(0), ray_hits_b)
    return ray_hits_b


def lidar_obs_dist(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, flatten: bool = False) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    if flatten:
        return sensor.data.distances
    return sensor.data.distances.unsqueeze(-1)


def lidar_obs_vel_rel(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """returns the velocity of the meshes of each ray hit position in the sensor's frame."""
    mesh_velocities_per_env, omega = obs_vel(env, sensor_cfg)
    pointcloud_mesh_ids = lidar_panoptic_segmentation(env, sensor_cfg, binary=False).squeeze()

    invalid_mask = pointcloud_mesh_ids == -1

    pointcloud_mesh_ids = pointcloud_mesh_ids.clone()
    pointcloud_mesh_ids[invalid_mask] = 0

    velocity_per_points = torch.gather(mesh_velocities_per_env, 1, pointcloud_mesh_ids.unsqueeze(-1).expand(-1, -1, 3))

    velocity_per_points[invalid_mask] = torch.tensor([0.0, 0.0, 0.0], device=velocity_per_points.device)

    return velocity_per_points


def lidar_obs_vel_norm(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """returns the norm of the velocity of the meshes of each ray hit position in the sensor's frame."""
    return torch.norm(lidar_obs_vel_rel(env, sensor_cfg), dim=-1)


def lidar_obs_vel_rel_heading(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """returns the relative heading angle.
    heading angle = angle between the velocity of the mesh (in (moving) body frame) of the ray hit and the ray direction.
    """
    points_2d = lidar_obs(env, sensor_cfg)[:, :, :2]
    velocities_2d = lidar_obs_vel_rel(env, sensor_cfg)[:, :, :2]

    dot_products = (points_2d * velocities_2d).sum(dim=2)
    norm_points = points_2d.norm(dim=2)
    norm_velocities = velocities_2d.norm(dim=2)
    cos_theta = dot_products / (norm_points * norm_velocities)

    # Mask where the norms product is close to zero
    mask_zero = torch.isclose(norm_points * norm_velocities, torch.tensor(0.0))

    # Set the cos_theta to 0 where the product of the norms is close to zero
    cos_theta[mask_zero] = 0

    # Calculate angles (in radians)
    angles = torch.acos(cos_theta.clamp(-1, 1))  # Clamping to ensure within the valid range for acos

    # Calculate the cross product z-component to determine the direction of angles
    cross_product_z = points_2d[:, :, 0] * velocities_2d[:, :, 1] - points_2d[:, :, 1] * velocities_2d[:, :, 0]

    # Adjust angles based on the direction indicated by the cross product
    adjusted_angles = torch.where(cross_product_z < 0, -angles, angles)

    # Wrap adjusted angles to [-pi, pi]
    wrapped_angles = math_utils.wrap_to_pi(adjusted_angles)

    return wrapped_angles


def lidar_obs_vel_rel_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, flatten: bool = False) -> torch.Tensor:
    """returns the relative velocity of the meshes of each ray hit position in the sensor's frame."""
    if flatten:
        return lidar_obs_vel_rel(env, sensor_cfg)[..., :2].reshape(env.num_envs, -1)

    return lidar_obs_vel_rel(env, sensor_cfg)[..., :2]


def lidar_obs_vel_b_static_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, flatten: bool = False) -> torch.Tensor:
    """returns the velocity of the meshes of each ray hit position in the static body frame."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    mesh_velocities_w: torch.Tensor = sensor.data.mesh_velocities_w
    mesh_velocities_rotated = math_utils.transform_points(
        mesh_velocities_w, quat=math_utils.quat_inv(sensor.data.quat_w)
    )[..., :2]

    pointcloud_mesh_ids = lidar_panoptic_segmentation(env, sensor_cfg, binary=False).squeeze()

    invalid_mask = pointcloud_mesh_ids == -1

    pointcloud_mesh_ids = pointcloud_mesh_ids.clone()
    pointcloud_mesh_ids[invalid_mask] = 0

    velocity_per_points = torch.gather(mesh_velocities_rotated, 1, pointcloud_mesh_ids.unsqueeze(-1).expand(-1, -1, 2))

    velocity_per_points[invalid_mask] = torch.tensor([0.0, 0.0], device=velocity_per_points.device)

    return velocity_per_points[..., :2].reshape(env.num_envs, -1) if flatten else velocity_per_points[..., :2]


# def lidar2d_dist_vel(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """returns the ray hit distance and velocity of the meshes of each ray hit position in the sensor's moving frame."""
#     return torch.cat((lidar_obs_dist(env, sensor_cfg).unsqueeze(-1), lidar_obs_vel_rel(env, sensor_cfg)[...,:2]), dim=2)


def obs_vel(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """velocity of the all meshes in all sensor frames (rot vel ignored)."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    mesh_velocities_w = sensor.data.mesh_velocities_w
    mesh_angular_velocities_w = sensor.data.mesh_angular_velocities_w

    robot_velocities_w = sensor.data.vel_w
    robot_angular_velocities_w = sensor.data.rot_vel_w

    mesh_velocities_shifted = mesh_velocities_w - robot_velocities_w.unsqueeze(1)
    mesh_angular_velocities_shifted = mesh_angular_velocities_w - robot_angular_velocities_w.unsqueeze(1)

    mesh_velocities_rotated = math_utils.transform_points(
        mesh_velocities_shifted, quat=math_utils.quat_inv(sensor.data.quat_w)
    )
    mesh_angular_velocities_rotated = math_utils.transform_points(
        mesh_angular_velocities_shifted, quat=math_utils.quat_inv(sensor.data.quat_w)
    )

    return mesh_velocities_rotated, mesh_angular_velocities_rotated


def lidar_privileged_mesh_pos_obs(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """privileged information related to the lidar sensor.
    Returns the mesh positions in the robot's frame."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    mesh_pos = sensor.data.mesh_positions_w[:, 1:, :]  # remove first mesh (ground)
    sensor_pos_w = sensor.data.pos_w
    sensor_quat_w = sensor.data.quat_w

    # mesh_orientations = sensor.data.mesh_orientations_w

    return transform_w_points_to_b_points(mesh_pos, sensor_pos_w, sensor_quat_w)


def obstacle_positions_sorted_flat(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, closest_N: int) -> torch.Tensor:
    """returns the closest N obstacle positions sorted by distance from the sensor.
    stacks the x and y together. Output shape is (num_envs, 2 * closest_N)."""

    closest_N = min(closest_N, env.num_envs)

    points = lidar_privileged_mesh_pos_obs(env, sensor_cfg)
    x_points, y_points, z_points = points[:, :, 0], points[:, :, 1], points[:, :, 2]

    distances = torch.sqrt(x_points**2 + y_points**2)

    closest_N_indices = torch.argsort(distances, dim=1)[:, :closest_N]

    x_points_sorted = torch.gather(x_points, 1, closest_N_indices)
    y_points_sorted = torch.gather(y_points, 1, closest_N_indices)
    # distances_sorted = torch.gather(distances, 1, closest_N_indices)

    return torch.concat((x_points_sorted, y_points_sorted), dim=1)


def lidar_panoptic_segmentation(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, flatten: bool = False, binary=True
) -> torch.Tensor:
    """privileged information related to the lidar sensor.
    returns the mesh index for each ray hit. -1 indicates no hit."""

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pan_seg_mask = sensor.data.ray_hit_mesh_idx

    undefined = torch.any(
        torch.stack(
            (
                torch.isinf(sensor.data.ray_hits_w[..., 0]),
                torch.isinf(sensor.data.ray_hits_w[..., 1]),
                torch.isinf(sensor.data.ray_hits_w[..., 2]),
            ),
            dim=0,
        ),
        dim=0,
    )

    seg_mask = torch.where(undefined.unsqueeze(2), torch.tensor(-1), pan_seg_mask)
    if flatten:
        seg_mask = seg_mask.squeeze()
    if binary:
        seg_mask[seg_mask > 0] = 1
    return seg_mask


"""
Actions.
"""


def last_low_level_action(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The last low-level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions


def second_last_low_level_action(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The second to last low level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.prev_low_level_actions


"""
Commands.
"""


def vel_commands(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The velocity command generated by the planner and given as input to the step function"""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.processed_actions


def generated_commands_reshaped(
    env: ManagerBasedRLEnv, command_name: str, unsqueeze_pos: int = 1, flatten: bool = False
) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name. Reshaped )"""
    if flatten:
        return env.command_manager.get_command(command_name)
    return env.command_manager.get_command(command_name).unsqueeze(unsqueeze_pos)


####################


import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

from ..actions import PerceptiveNavigationSE2Action
from ..wild_anymal_obs import ProprioceptiveObservation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def wild_anymal(env: ManagerBasedEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Wild anymal observation term."""

    # extract the used quantities (to enable type-hinting)
    term: NavigationSE2Action = env.action_manager._terms[action_term]
    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "wild_anymal_obs"):
        env.wild_anymal_obs = ProprioceptiveObservation(
            num_envs=env.num_envs,
            device=env.device,
            simulation_dt=env.physics_dt,
            control_dt=env.physics_dt * term.cfg.low_level_decimation,
        )

    env.wild_anymal_obs.update(
        robot.data.joint_pos[:, asset_cfg.joint_ids],
        robot.data.joint_vel[:, asset_cfg.joint_ids],
        term.processed_actions,
        robot.data.root_lin_vel_b,
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
    )
    return env.wild_anymal_obs.get_obs(use_raisim_order=True)


def cgp_state(env: ManagerBasedEnv):
    """Return the phase of the CPG as a state."""

    if not hasattr(env, "wild_anymal_obs"):
        return torch.zeros(env.num_envs, 8, device=env.device)
    else:
        phase = env.wild_anymal_obs.cpg.get_phase()
        return torch.cat([torch.sin(phase).view(-1, 4, 1), torch.cos(phase).view(-1, 4, 1)], dim=2).view(-1, 8)


# def wild_anymal(env: ManagerBasedEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Wild anymal observation term."""

#     if not hasattr(env, "wild_anymal_obs"):
#         env.wild_anymal_obs = ProprioceptiveObservation(
#             num_envs=env.num_envs,
#             device=env.device,
#             simulation_dt=env.physics_dt,
#             control_dt=env.physics_dt * env.cfg.low_level_decimation,
#         )

#     # extract the used quantities (to enable type-hinting)
#     term: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]
#     robot: Articulation = env.scene[asset_cfg.name]
#     env.wild_anymal_obs.update(
#         robot.data.joint_pos[:, asset_cfg.joint_ids],
#         robot.data.joint_vel[:, asset_cfg.joint_ids],
#         term.processed_actions,
#         robot.data.root_lin_vel_b,
#         robot.data.root_ang_vel_b,
#         robot.data.projected_gravity_b,
#     )
#     return env.wild_anymal_obs.get_obs(use_raisim_order=True)


"""
Robot Position
"""


def base_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root orientation in the asset's root frame.

    Args:
        env: The environment object.
        asset_cfg: The name of the asset.

    Returns:
        The root orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root position in the asset's root frame.

    Args:
        env: The environment object.
        asset_cfg: The name of the asset.

    Returns:
        The root position.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


"""
Metrics - used for performance analysis
"""


class EvalBuffer:
    def __init__(self) -> None:
        self.goal_reached_counter = 0
        # self.time_out_counter = 0
        self.collision_counter = 0
        self.dones_counter = 0
        self.episode_length_sum = 0
        self.success_rate = []

    def update(self, env: ManagerBasedRLEnv):
        try:
            goal_reached = env.termination_manager._term_dones["goal_reached"]
            # time_out = env.termination_manager._term_dones["time_out"]
            collision = env.termination_manager._term_dones["illegal_contact"]
            dones = env.termination_manager.dones

            self.goal_reached_counter += goal_reached.sum().item()
            # self.time_out_counter += time_out.sum().item()
            self.collision_counter += collision.sum().item()
            self.dones_counter += dones.sum().item()
            self.episode_length_sum += env.episode_length_buf[dones].sum().item()

            print(self.dones_counter)
            if env.common_step_counter % 100 == 0:
                self.success_rate.append(
                    0 if self.goal_reached_counter == 0 else self.goal_reached_counter / self.dones_counter
                )
                print(f"Success rate: {self.success_rate[-1]}")

            if env.common_step_counter == 500:
                self.goal_reached_counter = 0
                # self.time_out_counter = 0
                self.collision_counter = 0
                self.dones_counter = 0
                self.episode_length_sum = 0

            return torch.tensor([False]).repeat(env.num_envs, 1).to(env.device)
        except AttributeError:
            return torch.tensor([False]).repeat(env.num_envs, 1).to(env.device)


def metrics_timeout_signal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Timeout signal from the environment - meaning no collision.

    Args:
        env: The environment object.

    Returns:
        The timeout signal.
    """
    try:
        return env.termination_manager.time_outs.reshape(-1, 1)
    except AttributeError:
        return torch.tensor([False]).repeat(env.num_envs, 1).to(env.device)


def metrics_termination_signal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination signal from the environment - meaning collision.

    Args:
        env: The environment object.

    Returns:
        The termination signal.
    """
    try:
        return env.termination_manager.terminated.reshape(-1, 1)
    except AttributeError:
        return torch.tensor([False]).repeat(env.num_envs, 1).to(env.device)


def metrics_dones_signal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dones signal from the environment - meaning collision or timeout.

    Args:
        env: The environment object.

    Returns:
        The dones signal.
    """
    try:
        return env.termination_manager.dones.reshape(-1, 1)
    except AttributeError:
        return torch.tensor([False]).repeat(env.num_envs, 1).to(env.device)


def metrics_goal_reached(
    env: ManagerBasedRLEnv, distance_threshold: float = 0.5, speed_threshold: float = 0.05
) -> torch.Tensor:
    """Goal reached signal from the environment.

    Args:
        env: The environment object.
        distance_threshold: The distance threshold to the goal.
        speed_threshold: The speed threshold at the goal.

    Returns:
        The goal reached signal.
    """
    try:
        return mdp.goal_reached(env, distance_threshold=distance_threshold, speed_threshold=speed_threshold).reshape(
            -1, 1
        )
    except AttributeError:
        return torch.tensor([False]).repeat(env.num_envs, 1).to(env.device)


def metrics_goal_position(env: ManagerBasedRLEnv, command_name: str = "robot_goal") -> torch.Tensor:
    """Goal position in the world frame

    Args:
        env: The environment object.

    Returns:
        The goal position.
    """
    try:
        goal_cmd_geneator: mdp.RobotGoalCommand = env.command_manager._terms[command_name]
        return goal_cmd_geneator.pos_command_w[:, 0:2].reshape(-1, 2)
    except AttributeError:
        return torch.tensor([0.0, 0.0]).repeat(env.num_envs, 1).to(env.device)


def metrics_start_position(env: ManagerBasedRLEnv, command_name: str = "robot_goal") -> torch.Tensor:
    """Start position in the world frame

    Args:
        env: The environment object.

    Returns:
        The start position.
    """
    try:
        goal_cmd_geneator: mdp.RobotGoalCommand = env.command_manager._terms[command_name]
        return goal_cmd_geneator.pos_spawn_w[:, 0:2].reshape(-1, 2)
    except AttributeError:
        return torch.tensor([0.0, 0.0]).repeat(env.num_envs, 1).to(env.device)


def metrics_robot_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Robot position in the world frame

    Args:
        env: The environment object.
        asset_cfg: The name of the asset.

    Returns:
        The robot position.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0:2]


def metrics_path_length(env: ManagerBasedRLEnv, command_name: str = "robot_goal") -> torch.Tensor:
    """Path length

    Args:
        env: The environment object.

    Returns:
        The path length.
    """
    try:
        goal_cmd_geneator: mdp.RobotGoalCommand = env.command_manager._terms[command_name]
        return goal_cmd_geneator.path_length_command.reshape(-1, 1)
    except AttributeError:
        return torch.tensor([0.0]).repeat(env.num_envs, 1).to(env.device)


def metrics_undesired_contacts(env: ManagerBasedEnv, threshold: float = 0.5, body_names: list = None) -> torch.Tensor:
    """Undesired contacts

    Args:
        env: The environment object.
        threshold: The threshold for the contact force.
        body_names: The names of the bodies to check for contact.

    Returns:
        The undesired contacts.
    """
    try:
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=body_names)
        sensor_cfg.resolve(env.scene)
        return mdp.undesired_contacts(env, threshold=threshold, sensor_cfg=sensor_cfg).reshape(-1, 1)
    except AttributeError:
        return torch.tensor([0.0]).repeat(env.num_envs, 1).to(env.device)


def metrics_undesired_contacts_wheeled_robot(
    env: ManagerBasedEnv, threshold: float = 0.5, body_names: list = None, body_names_wheels: list = None
) -> torch.Tensor:
    """Undesired contacts for wheeled robots

    Args:
        env: The environment object.
        threshold: The threshold for the contact force.
        body_names: The names of the bodies to check for contact.
        body_names_wheels: The names of the wheels to check for contact.

    Returns:
        The undesired contacts.
    """
    try:
        # Body contacts
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=body_names)
        sensor_cfg.resolve(env.scene)
        undesired_contacts = mdp.undesired_contacts(env, threshold=threshold, sensor_cfg=sensor_cfg).reshape(-1, 1)
        # Wheel contacts
        sensor_cfg_wheels = SceneEntityCfg("contact_forces", body_names=body_names_wheels)
        sensor_cfg_wheels.resolve(env.scene)
        undesired_wheel_contacts = mdp.undesired_wheel_contacts(
            env, threshold=threshold, sensor_cfg=sensor_cfg_wheels
        ).reshape(-1, 1)
        return undesired_contacts + undesired_wheel_contacts
    except AttributeError:
        return torch.tensor([0.0]).repeat(env.num_envs, 1).to(env.device)


def metrics_episode_length(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode length in steps (1 step is 0.1s)

    Args:
        env: The environment object.

    Returns:
        The episode length."""
    try:
        return env.episode_length_buf.reshape(-1, 1).to(env.device)
    except AttributeError:
        return torch.tensor([0]).repeat(env.num_envs, 1).to(env.device)
