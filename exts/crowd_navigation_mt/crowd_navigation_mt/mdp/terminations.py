from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from crowd_navigation_mt.terrains import SemanticTerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from crowd_navigation_mt.mdp import RobotGoalCommand


"""
MDP terminations.
"""


def in_restricted_area(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        area_label_idx: list[int] = [4],
):
    """check how the semantic map is created in crowd_navigation_mt/terrains/elevation_map/semantic_height_map.py
    or in the "SemanticTerrainImporter" for the label indices"""
    label_idx: torch.tensor = torch.tensor(area_label_idx).to(device=env.device)
    robot: Articulation = env.scene[asset_cfg.name]
    if isinstance(env.scene.terrain, SemanticTerrainImporter):
        terrain = env.scene.terrain
        grid_map = terrain.grid_map
        state_to_grid = (robot.data.root_pos_w[:, :2] - terrain.transform_vector) / terrain.cfg.semantic_terrain_resolution
        grid_to_idx = state_to_grid.int()
        termination = torch.isin(grid_map[grid_to_idx[:, 0], grid_to_idx[:, 1]], label_idx)
        # if termination.any():
        #     print("[INFO]: Robot is in restricted area.")
        return termination
    
    else:
        raise TypeError(
            f"Expected 'SemanticTerrainImporter' as env.scene.terrain, but got '{type(env.scene.terrain).__name__}' instead."
        )



def at_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    speed_threshold: float = 0.05,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        speed_threshold: The speed threshold at the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[goal_cmd_name]
    # check for termination
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    abs_velocity = torch.norm(asset.data.root_vel_w[:, 0:6], dim=1, p=2)
    # Check conditions
    within_goal = distance_goal < distance_threshold
    within_speed = abs_velocity < speed_threshold
    # # Ugly Fix for data logging (otherwise goal reached is always 0), so it stays at goal for 1 step longer
    # goal_reached_obs = env.observation_manager._obs["metrics"]["goal_reached"] == 1.0
    # Return termination
    at_goal = torch.logical_and(within_goal, within_speed)
    return at_goal


def update_command_on_termination(
    env: ManagerBasedRLEnv,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    Args:
        env: The learning environment.
        goal_cmd_name: The name of the command generator.

    Returns:
        Boolean tensor, always False.
    """
    goal_cmd_geneator: RobotGoalCommand = env.command_manager._terms[goal_cmd_name]

    failure = env.termination_manager._term_dones["base_contact"] | env.termination_manager._term_dones["thigh_contact"]

    goal_cmd_geneator.update_success(env.termination_manager._term_dones["goal_reached"])
    goal_cmd_geneator.update_failures(failure)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def illegal_wheel_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold. Since we consider here wheel contacts,
    we only check the contact force in the x-y plane.

    Args:
        env: The learning environment.
        threshold: The force threshold.
        sensor_cfg: The name of the sensor.

    Returns:
        Boolean tensor indicating whether the contact force exceeds the threshold.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=-1), dim=1)[0] > threshold, dim=1
    )
