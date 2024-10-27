# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from crowd_navigation_mt.mdp import (
        RobotGoalCommand,
        Uniform2dCoord,
        DirectionCommand,
        ObstacleActionTermSimple,
    )


def terrain_levels_distance(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # cmd_generator: DirectionCommand = env.command_manager._terms["robot_direction"]
    cmd_generator: DirectionCommand = env.command_manager._terms["robot_direction"]
    # compute the distance the robot walked
    # distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    distance_vec = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = torch.sum(cmd_generator.direction_command[env_ids, :2] * distance_vec, dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0]
    # robots that collided without reaching the required distance go to simpler terrains
    move_down = env.termination_manager._term_dones["illegal_contact"][env_ids] & ~move_up
    # update the metrics
    # cmd_generator.metrics["move_up"] *= 0
    # cmd_generator.metrics["move_down"] *= 0
    # cmd_generator.metrics["move_up"][env_ids] += move_up.float().mean()
    # cmd_generator.metrics["move_down"][env_ids] += move_down.float().mean()
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # TODO randomly move down, such that the robots dont forget the easier terrains

    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


#########################################################################################################
# attribute needs to be outside of the function for some reasons
#########################################################################################################

def terrain_levels(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        dist_threshold: float = 0.5,
        required_successes: int = 8,
        required_failures: int = 8,
        move_down_prob: float = 0.05,
    ) -> torch.Tensor:
    """Curriculum based on wether the goal was reached or not.

    This term is used to increase the difficulty of the terrain when the robot reaches the goal.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter | None = env.scene.terrain
    # cmd_generator: DirectionCommand = env.command_manager._terms["robot_direction"]
    cmd_generator: RobotGoalCommand = env.command_manager._terms["robot_goal"]
    # compute the distance the robot walked
    # distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # distance_vec = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]

    error_vec = cmd_generator.pos_command_w[env_ids, :2] - asset.data.root_pos_w[env_ids, :2]
    at_goal = torch.norm(error_vec, dim=1) < dist_threshold
    goal_reached = cmd_generator.goal_dist_increment[env_ids] > 1
    goal_reached |= env.termination_manager._term_dones["goal_reached"][env_ids]

    # robots that walked far enough progress to harder terrains
    # move_up = torch.norm(error_vec, dim=1) > terrain.cfg.terrain_generator.size[0]
    # robots that collided without reaching the required distance go to simpler terrains
    increment_move_up = at_goal | goal_reached
    increment_move_down = env.termination_manager.terminated[env_ids] & ~increment_move_up

    # if self.move_up_counter is None or self.move_down_counter is None:
    #     self.move_up_counter = torch.zeros(env.num_envs, device=env.device)
    #     self.move_down_counter = torch.zeros(env.num_envs, device=env.device)
    if self.move_plus_counter is None:
        self.move_plus_counter = torch.zeros(env.num_envs, device=env.device)

    # self.move_up_counter[env_ids] += increment_move_up.int()
    # self.move_down_counter[env_ids] += increment_move_down.int()
    self.move_plus_counter[env_ids] += increment_move_up.int() - increment_move_down.int()

    # move_up = self.move_up_counter[env_ids] > required_successes
    # move_down = self.move_down_counter[env_ids] > required_failures
    move_up = self.move_plus_counter[env_ids] > required_successes
    move_down = self.move_plus_counter[env_ids] < -required_failures

    # self.move_up_counter[env_ids][move_up] = 0
    # self.move_down_counter[env_ids][move_down] = 0
    self.move_plus_counter[env_ids[move_up | move_down]] = 0

    # randomly move down instead of up
    random_move_down = (torch.rand_like(move_up.float()) < move_down_prob) & move_up
    move_up = move_up & ~random_move_down
    move_down = move_down | random_move_down

    # update the metrics
    # cmd_generator.metrics["move_up"] *= 0
    # cmd_generator.metrics["move_down"] *= 0
    # cmd_generator.metrics["move_up"][env_ids] += move_up.float().mean()
    # cmd_generator.metrics["move_down"][env_ids] += move_down.float().mean()
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


class TerrainLevelsDistance:
    def __init__(self):
        # self.move_up_counter: torch.Tensor | None = None
        # self.move_down_counter: torch.Tensor | None = None
        self.move_plus_counter: torch.Tensor | None = None

    def terrain_levels(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        dist_threshold: float = 0.5,
        required_successes: int = 8,
        required_failures: int = 8,
        move_down_prob: float = 0.05,
    ) -> torch.Tensor:
        """Curriculum based on wether the goal was reached or not.

        This term is used to increase the difficulty of the terrain when the robot reaches the goal.

        .. note::
            It is only possible to use this term with the terrain type ``generator``. For further information
            on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

        Returns:
            The mean terrain level for the given environment ids.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        terrain: TerrainImporter | None = env.scene.terrain
        # cmd_generator: DirectionCommand = env.command_manager._terms["robot_direction"]
        cmd_generator: RobotGoalCommand = env.command_manager._terms["robot_goal"]
        # compute the distance the robot walked
        # distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
        # distance_vec = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]

        error_vec = cmd_generator.pos_command_w[env_ids, :2] - asset.data.root_pos_w[env_ids, :2]
        at_goal = torch.norm(error_vec, dim=1) < dist_threshold
        goal_reached = cmd_generator.goal_dist_increment[env_ids] > 1
        goal_reached |= env.termination_manager._term_dones["goal_reached"][env_ids]

        # robots that walked far enough progress to harder terrains
        # move_up = torch.norm(error_vec, dim=1) > terrain.cfg.terrain_generator.size[0]
        # robots that collided without reaching the required distance go to simpler terrains
        increment_move_up = at_goal | goal_reached
        increment_move_down = env.termination_manager.terminated[env_ids] & ~increment_move_up

        # if self.move_up_counter is None or self.move_down_counter is None:
        #     self.move_up_counter = torch.zeros(env.num_envs, device=env.device)
        #     self.move_down_counter = torch.zeros(env.num_envs, device=env.device)
        if self.move_plus_counter is None:
            self.move_plus_counter = torch.zeros(env.num_envs, device=env.device)

        # self.move_up_counter[env_ids] += increment_move_up.int()
        # self.move_down_counter[env_ids] += increment_move_down.int()
        self.move_plus_counter[env_ids] += increment_move_up.int() - increment_move_down.int()

        # move_up = self.move_up_counter[env_ids] > required_successes
        # move_down = self.move_down_counter[env_ids] > required_failures
        move_up = self.move_plus_counter[env_ids] > required_successes
        move_down = self.move_plus_counter[env_ids] < -required_failures

        # self.move_up_counter[env_ids][move_up] = 0
        # self.move_down_counter[env_ids][move_down] = 0
        self.move_plus_counter[env_ids[move_up | move_down]] = 0

        # randomly move down instead of up
        random_move_down = (torch.rand_like(move_up.float()) < move_down_prob) & move_up
        move_up = move_up & ~random_move_down
        move_down = move_down | random_move_down

        # update the metrics
        # cmd_generator.metrics["move_up"] *= 0
        # cmd_generator.metrics["move_down"] *= 0
        # cmd_generator.metrics["move_up"][env_ids] += move_up.float().mean()
        # cmd_generator.metrics["move_down"][env_ids] += move_down.float().mean()
        # update terrain levels
        terrain.update_env_origins(env_ids, move_up, move_down)

        # return the mean terrain level
        return torch.mean(terrain.terrain_levels.float())


def modify_goal_distance(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # new_distance: float,
    command_name: str = "robot_goal",
    step_size: float = 0.5,
    required_successes: int = 2,
):
    """Curriculum that modifies the goal distance.

    Args:
        command_name: The name of the command generator.
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        new_distance: The new distance of the goal.

    Returns:
        The new distance of the goal.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_generator: RobotGoalCommand = env.command_manager._terms[command_name]
    # update the goal distance
    goal_cmd_generator.update_goal_distance(increase_by=step_size, required_successes=required_successes)
    return goal_cmd_generator.goal_dist.mean()


def modify_goal_distance_relative(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # new_distance: float,
    command_name: str = "robot_goal",
    step_size: int = 1,
    required_successes: int = 2,
    max_distance: int = 10,
):
    """Curriculum that modifies the goal distance by multiplying the original one with an integer.
    Useful for gridworlds.

    Args:
        command_name: The name of the command generator.
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        new_distance: The new distance of the goal.
        max_distance: The maximum relative distance of the goal (max_dist * base dist).

    Returns:
        The new distance of the goal.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_generator: RobotGoalCommand = env.command_manager._terms[command_name]
    # update the goal distance
    goal_cmd_generator.update_goal_distance_rel(
        increase_by=step_size, required_successes=required_successes, max_goal_dist=max_distance
    )
    return goal_cmd_generator.goal_dist_rel.float().mean().item()


def modify_goal_distance_relative_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # new_distance: float,
    command_name: str = "robot_goal",
    start_size: int = 1,
    end_size: int = 2,
    start_step: int = 0,
    num_steps: int = 10000,
):
    """Curriculum that modifies the goal distance by multiplying the original one with an integer.
    Useful for gridworlds.

    Args:
        command_name: The name of the command generator.
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        start_size: The initial relative distance of the goal.
        end_size: The final relative distance of the goal.
        start_step: The step at which the curriculum starts.
        num_steps: The number of steps to reach the final distance.

    Returns:
        The new distance of the goal.
    """

    step_size_float = start_size + min(max(env.common_step_counter - start_step, 0) / num_steps, 1) * (
        end_size - start_size
    )
    step_size = int(step_size_float)
    # extract the used quantities (to enable type-hinting)
    goal_cmd_generator: RobotGoalCommand = env.command_manager._terms[command_name]
    # update the goal distance
    goal_cmd_generator.goal_dist_rel[:] = step_size
    return step_size_float


def modify_obstacle_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # new_distance: float,
    action_name: str = "obstacle_bot_positions",
    start_velocity: float = 0.0,
    start_step: int = 0,
    end_velocity: float = 1.0,
    num_steps: int = 1000,
):
    """Curriculum that modifies the velocity of the obstacles.

    Args:
        command_name: The name of the command generator.
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        start_velocity: The initial velocity of the obstacles.
        start_step: The step at which the curriculum starts.
        end_velocity: The final velocity of the obstacles.
        num_steps: The number of steps to reach the final velocity.

    Returns:
        The obstacle max velocity.
    """

    velocity = start_velocity + min(max(env.common_step_counter - start_step, 0) / num_steps, 1) * (
        end_velocity - start_velocity
    )

    # extract the used quantities (to enable type-hinting)
    obstacle_action: ObstacleActionTermSimple = env.action_manager._terms[action_name]
    # update the goal distance
    obstacle_action.max_velocity = velocity
    return velocity


class EnvSpacingCurriculum:

    original_grid: torch.Tensor = None
    original_spacing: float = None

    @classmethod
    def scale_env_spacing_linearly(
        cls,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        initial_dist_scale: float,
        final_dist_scale: float,
        start_step: int,
        end_step: int,
        obstacle_command_name: str = "obstacle_target_pos",
    ):
        """Curriculum that modifies a reward weight linearly.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            initial dist: Env spacing at the beginning of the curriculum.
            final dist: Env spacing at the end of the curriculum.
            start_step: The step at which the curriculum starts.
            end_step: The step at which the curriculum ends.

        Returns:
            The new weight of the reward term.

        NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
        """
        # compute the new weight
        num_steps = end_step - start_step
        scale = (
            initial_dist_scale
            + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
            * (final_dist_scale - initial_dist_scale)
            / num_steps
        )

        if cls.original_grid is None:
            cls.original_grid = env.scene.env_origins[:, :2].clone()
            cls.original_spacing = env.cfg.scene.env_spacing

        # update term settings robots
        env.scene.env_origins[:, :2] = cls.original_grid * scale
        env.cfg.scene.env_spacing = cls.original_spacing * scale
        # and obstacles
        obstacle_cmd: Uniform2dCoord = env.command_manager._terms[obstacle_command_name]
        obstacle_cmd.env = env

        return scale


def modify_reward_weight_linearly(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_step: int,
    end_step: int,
):
    """Curriculum that modifies a reward weight linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        initial weight: The weight of the reward term at the beginning of the curriculum.
        final weight: The weight of the reward term at the end of the curriculum.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends.

    Returns:
        The new weight of the reward term.

    NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
    """
    # obtain term settings
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # compute the new weight
    num_steps = end_step - start_step
    weight = (
        initial_weight
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (final_weight - initial_weight)
        / num_steps
    )
    # update term settings
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(weight)


def modify_heading_randomization_linearly(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_perturbation: float,
    final_perturbation: float,
    start_step: int,
    end_step,
):
    """Curriculum that modifies a randomization parameter linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the randomization term.
        initial_perturbation: The perturbation of the randomization term at the beginning of the curriculum.
        final_perturbation: The perturbation of the randomization term at the end of the curriculum.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends.

    Returns:
        The new perturbation of the randomization term.

    NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
    """
    # obtain term settings
    term_cfg = env.randomization_manager.get_term_cfg(term_name)
    # compute the new weight
    num_steps = end_step - start_step
    perturbation = (
        initial_perturbation
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (final_perturbation - initial_perturbation)
        / num_steps
    )
    # update term settings
    term_cfg.params["additive_heading_range"]["yaw"] = (-perturbation, perturbation)
    env.randomization_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(perturbation)
