# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.assets import Articulation

from crowd_navigation_mt import mdp
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from crowd_navigation_mt.mdp.commands import RobotGoalCommand, DirectionCommand


def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


"""FROM IMRL PLANNER"""

"""
Navigation rewards.
"""


def goal_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    speed_threshold: float = 0.05,
    command_name: str = "robot_goal",
) -> torch.Tensor:
    """Reward the agent for reaching the goal.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        speed_threshold: The speed threshold at the goal.

    Returns:
        Sparse reward of 1 if the goal is reached, 0 otherwise.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # compute the reward
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    abs_velocity = torch.norm(asset.data.root_vel_w[:, 0:3], dim=1, p=2)
    reward = torch.where(
        distance_goal < distance_threshold, torch.ones_like(distance_goal), torch.zeros_like(distance_goal)
    )

    env_ids = torch.nonzero(reward).flatten()

    # goal_cmd_geneator.increment_goal_distance(env_ids)
    goal_cmd_geneator._resample_command(env_ids)

    reward = torch.where(abs_velocity < speed_threshold, reward, torch.zeros_like(abs_velocity))

    return reward


def goal_closeness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "robot_goal",
) -> torch.Tensor:
    """Reward for getting closer to the goal, linearly from 0 to 1.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Sparse reward of 1 if the goal is reached, 0 otherwise.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    max_dist = goal_cmd_geneator.goal_dist

    # compute the reward
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    # abs_velocity = torch.norm(asset.data.root_vel_w[:, 0:3], dim=1, p=2)
    rel_dist = distance_goal / max_dist
    return 1 - rel_dist


def obstacle_distance(
    env: ManagerBasedRLEnv,
    # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1,
    dist_std: float = 1,
    dist_sensor: SceneEntityCfg = SceneEntityCfg("lidar"),
):
    """Reward the agent for avoiding obstacles using L2-Kernel.

    Args:
        env: The learning environment.
        threshold: The distance threshold to the obstacles.
        dist_std: The standard deviation of the distance to the obstacles.
        dist_sensor: The name of the distance sensor (2d lidar).

    Returns:
        Dense reward [0, +1] based on the distance to the obstacles. Needs to have negative weight.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: RayCaster = env.scene.sensors[dist_sensor.name]
    distances = contact_sensor.data.distances
    valid = distances > 1e-3
    filtered_data = torch.where(valid, distances, torch.tensor(float("inf")))
    min_values = torch.min(filtered_data, dim=1)[0]

    reward = 1 - torch.tanh((min_values - threshold) / dist_std)
    return reward


def obstacle_distance_in_front(
    env: ManagerBasedRLEnv,
    # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1,
    dist_std: float = 1,
    degrees: float = 30,
    dist_sensor: SceneEntityCfg = SceneEntityCfg("lidar"),
):
    """Reward the agent for avoiding obstacles using L2-Kernel.

    Args:
        env: The learning environment.
        threshold: The distance threshold to the obstacles.
        dist_std: The standard deviation of the distance to the obstacles.
        degrees: degrees in front of the robot to consider.
        dist_sensor: The name of the distance sensor (2d lidar).

    Returns:
        Dense reward [0, +1] based on the distance to the obstacles. Needs to have negative weight.
    """
    # extract the used quantities (to enable type-hinting)

    angle_threshold = torch.tensor(degrees / 2 * 3.141592653589793 / 180.0)
    # contact_sensor: RayCaster = env.scene.sensors[dist_sensor.name]
    points = mdp.lidar_obs(env, dist_sensor, True)[:, :, :2]
    invalid_dist = torch.isclose(points[..., 1], torch.tensor(0.0)) | torch.isclose(points[..., 0], torch.tensor(0.0))
    angles = torch.atan2(points[..., 1], points[..., 0])
    valid_angles = torch.abs(angles) < angle_threshold
    valid = ~invalid_dist & valid_angles

    filtered_data = torch.where(valid, torch.norm(points, dim=2), torch.tensor(float("inf")))
    min_values = torch.min(filtered_data, dim=1)[0]

    # env.observation_manager.compute_group(group_name="policy").shape

    reward = 1 - torch.tanh((min_values - threshold) / dist_std)
    return reward


def goal_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    command_name: str = "robot_goal",
    normalize_distance: bool = True,
) -> torch.Tensor:
    """Reward the agent for making progress towards the goal based on dot product between displacement vector and velocity vector.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        threshold: The distance threshold to the goal.

    Returns:
        Dense reward based on the dot product between displacement vector direction and velocity vector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # compute the reward
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    displacement_vector = (
        goal_cmd_geneator.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2]
    ) / distance_goal.unsqueeze(-1)

    if normalize_distance:
        displacement_vector = displacement_vector / (torch.norm(displacement_vector, dim=1).unsqueeze(-1) + 1e-6)

    reward = torch.sum(displacement_vector * asset.data.root_vel_w[:, :2], dim=1)
    reward = torch.clip(reward, min=0, max=2)
    # check if goal reached
    # reward = torch.where(distance_goal < threshold, 2 * torch.ones_like(distance_goal), reward)
    return reward


def progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "robot_direction",
) -> torch.Tensor:
    """Reward the agent for making progress towards based on dot product between direction vector and velocity vector.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        threshold: The distance threshold to the goal.

    Returns:
        Dense reward [-0.5, 1] based on the dot product between commanded direction vector and velocity vector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: DirectionCommand = env.command_manager._terms[command_name]
    # compute the reward
    reward = torch.sum(goal_cmd_geneator.direction_command[:, :2] * asset.data.root_vel_w[:, :2], dim=1)
    reward = torch.clip(reward, min=-0.5, max=1)
    # check if goal reached
    return reward


def stage_cleared(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sparse reward for clearing a terrain stage."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    cmd_generator: DirectionCommand = env.command_manager._terms["robot_direction"]
    # compute the distance the robot walked
    # distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    distance_vec = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    distance = torch.sum(cmd_generator.direction_command[:, :2] * distance_vec, dim=1)
    # robots that walked far enough progress to harder terrains

    move_up = distance > terrain.cfg.terrain_generator.size[0]
    env_ids = torch.where(move_up)[0]
    move_down = ~move_up[env_ids]
    terrain.update_env_origins(env_ids, move_up[env_ids], move_down)

    return move_up.float()


def waypoint_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    command_name: str = "robot_goal",
) -> torch.Tensor:
    """Reward the agent for making progress towards the next waypoint.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        threshold: The distance threshold to the goal.

    Returns:
        Dense reward [0, 0.5] based on the dot product between displacement vector and velocity vector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # compute the reward
    distance_wp = torch.norm(
        asset.data.root_pos_w[:, :2] - mdp.high_level_navigation_waypoints(env).reshape(env.num_envs, -1, 3)[:, 0, :2],
        dim=1,
        p=2,
    )
    displacement_vector = (
        mdp.high_level_navigation_waypoints(env).reshape(env.num_envs, -1, 3)[:, 0, :2] - asset.data.root_pos_w[:, :2]
    ) / distance_wp.unsqueeze(-1)
    reward = torch.sum(displacement_vector * torch.abs(asset.data.root_vel_w[:, :2]), dim=1)
    reward = torch.clip(reward, min=0, max=0.5)
    # check if goal reached
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    reward = torch.where(distance_goal < threshold, 0.5 * torch.ones_like(distance_goal), reward)
    return reward


def waypoint_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    command_name: str = "robot_goal",
):
    """Reward the agent for following the direction of the waypoints using Cosine Similarity.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        threshold: The distance threshold to the goal.

    Returns:
        Dense reward [-1, +1] based on the cosine similarity between the heading vector and the displacement vector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # compute the reward
    next_wp = mdp.high_level_navigation_waypoints(env).reshape(env.num_envs, -1, 3)[:, 0, :2]
    heading_error = torch.atan2(next_wp[:, 1], next_wp[:, 0])  # [-pi, +pi], in robot frame
    reward = torch.cos(heading_error)  # based on cosine similarity (+1, -1)
    # check if goal reached
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    reward = torch.where(distance_goal < threshold, 1.0 * torch.ones_like(distance_goal), reward)
    return reward


def near_goal_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    command_name: str = "robot_goal",
) -> torch.Tensor:
    """Reward the agent for being stable near the goal.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        threshold: The distance threshold to the goal.

    Returns:
        Dense reward [0, +1] based on the distance to the goal and the velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]
    # compute the reward
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    square_velocity = torch.norm(asset.data.root_vel_w[:, 0:6], dim=1, p=2) ** 2
    reward = torch.where(
        distance_goal < threshold, torch.ones_like(distance_goal), torch.zeros_like(distance_goal)
    ) * torch.exp(-2.0 * square_velocity)
    return reward


def no_robot_movement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_distance_thresh: float = 0.75,
    command_name: str = "robot_goal",
) -> torch.Tensor:
    """Penalize the agent for staying at the same location for multiple timesteps.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, -1] based on the robot's velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[command_name]

    # compute the pentaly for staying at the same location
    position_history = env.observation_manager.last_obs["metrics"]["robot_position_history"].reshape(
        env.num_envs, -1, 2
    )
    differences = position_history[:, 1:] - position_history[:, :-1]  # Differences between consecutive points
    distances = torch.sqrt(torch.sum(differences**2, axis=2))  # Euclidean distance for each pair
    sum_distances = torch.sum(distances, axis=1)

    # Reward
    penalty = torch.exp(
        -40 * sum_distances
    )  # Exponential decay of the penalty - expect above 20cm robot to move for 0 penalty

    # Check if goal reached
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    reward = torch.where(distance_goal < goal_distance_thresh, torch.zeros_like(distance_goal), penalty)

    return reward


def no_robot_movement_2d(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize the agent for staying at the same location for multiple timesteps.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, -1] based on the robot's velocity.
    """

    # compute the pentaly for staying at the same location
    position_history = env.observation_manager.last_obs["metrics"]["robot_position_history"].reshape(
        env.num_envs, -1, 2
    )
    differences = position_history[:, 1:] - position_history[:, :-1]  # Differences between consecutive points
    distances = torch.sqrt(torch.sum(differences**2, axis=2))  # Euclidean distance for each pair
    sum_distances = torch.sum(distances, axis=1)

    # Reward
    penalty = torch.exp(
        -10 * sum_distances
    )  # Exponential decay of the penalty - expect above 20cm robot to move for 0 penalty

    # Check if goal reached
    reward = penalty

    return reward


"""
Action rewards.
"""


def backwards_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for moving backwards using L2-Kernel

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the backward velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    backward_movement_idx = torch.where(
        forward_velocity < 0.0, torch.ones_like(forward_velocity), torch.zeros_like(forward_velocity)
    )
    reward = torch.square(backward_movement_idx * forward_velocity)
    reward = torch.clip(reward, min=0, max=1.0)
    return reward


def lateral_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward the agent for moving lateral using L2-Kernel

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    reward = torch.square(lateral_velocity)
    reward = torch.clip(reward, min=0, max=1.0)
    return reward


"""
Contact sensor
"""


def undesired_wheel_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold.

    Args:
        env: The learning environment.
        threshold: The contact force threshold.
        sensor_cfg: The name of the contact sensor.

    Returns:
        Dense reward [0, +n] based on the number of undesired contacts."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)
