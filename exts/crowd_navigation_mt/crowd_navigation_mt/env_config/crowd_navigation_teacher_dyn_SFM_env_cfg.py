# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING


from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm


import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg


from .crowd_navigation_dyn_obs_base_env_cfg import CrowdNavigationEnvCfg

from crowd_navigation_mt.mdp import LidarHistoryTermCfg

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm


import omni.isaac.lab.sim as sim_utils

##
# Pre-defined configs
##
from crowd_navigation_mt.assets.simple_obstacles import (
    OBS_CFG,
    EMPTY_OBS_CFG,
    CYLINDER_HUMANOID_CFG,
    MY_RAY_CASTER_MARKER_CFG,
    WALL_CFG,
)
from crowd_navigation_mt.mdp import ObservationHistoryTermCfg


# observation group:
@configclass
class TeacherPolicyObsCfg(ObsGroup):
    """Observations for the teacher policy"""

    # commands
    target_position = ObsTerm(
        func=mdp.generated_commands_reshaped, params={"command_name": "robot_goal", "flatten": True}
    )

    # add cpg state to the proprioception group
    cpg_state = ObsTerm(func=mdp.cgp_state)


    lidar_distances_history = LidarHistoryTermCfg(
        func=mdp.LidarHistory,
        # params={"kwargs" : {"method": "get_history"}},
        params={"method": "get_history"},
        history_length=5, 
        decimation=1, 
        sensor_cfg=SceneEntityCfg("lidar"), 
        return_pose_history=True,
    )

    robot_position_history = ObservationHistoryTermCfg(
            func=mdp.ObservationHistory,
            params={"method": "get_history_of_positions"},
            # params={"kwargs" : {"method": "get_history_of_positions"}},
            history_length_actions=10,
            history_length_positions=10,
            history_time_span_positions=5,
            history_time_span_actions=5
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TeacherTerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*THIGH"]),  # , ".*THIGH"
            "threshold": 1.0,
        },
    )

    goal_reached = DoneTerm(
        func=mdp.at_goal,  # reward is high to compensate for reset penalty
        params={"distance_threshold": 0.5, "speed_threshold": 2.5},
    )


@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP."""

    # -- tasks
    goal_reached = RewTerm(
        func=mdp.goal_reached,  # reward is high to compensate for reset penalty
        weight=50.0,  # Sparse Reward of {0.0,0.2} --> Max Episode Reward: 2.0
        params={"distance_threshold": 0.5, "speed_threshold": 2.5},
    )

    goal_progress = RewTerm(
        func=mdp.goal_progress,
        weight=2,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    goal_closeness = RewTerm(
        func=mdp.goal_closeness,
        weight=1,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    # # -- penalties

    # lateral_movement = RewTerm(
    #     func=mdp.lateral_movement,
    #     weight=-0.01,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    # )

    # backward_movement = RewTerm(
    #     func=mdp.backwards_movement,
    #     weight=-0.01,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    # )

    # episode_termination = RewTerm(
    #     func=mdp.is_terminated,
    #     weight=-500.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    # )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, weight=-0.1  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )

    # no_robot_movement = RewTerm(
    #     func=mdp.no_robot_movement_2d,
    #     weight=-5,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    # )

    #  penalty for being close to the obstacles
    close_to_obstacle = RewTerm(
        func=mdp.obstacle_distance,
        weight=-2.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 1.5, "dist_std": 0.5, "dist_sensor": SceneEntityCfg("lidar")},
    )

    obstacle_in_front_narrow = RewTerm(
        func=mdp.obstacle_distance_in_front,
        weight=-1.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 2, "dist_std": 1.5, "dist_sensor": SceneEntityCfg("lidar"), "degrees": 30},
    )

    obstacle_in_front_wide = RewTerm(
        func=mdp.obstacle_distance_in_front,
        weight=-1.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 1, "dist_std": 0.5, "dist_sensor": SceneEntityCfg("lidar"), "degrees": 60},
    )

    # far_from_obstacle = RewTerm(
    #     func=mdp.obstacle_distance,
    #     weight=-0.1,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    #     params={"threshold": 1, "dist_std": 5, "dist_sensor": SceneEntityCfg("lidar")},
    # )
    # TODO add penality for obstacles being in front of the robot

    # # penalty for colliding with obstacles
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-200.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", "base"]), # ".*HIP", ".*SHANK",
    #         "threshold": 0.5,
    #     },
    # )

    # # reward for being alive
    # is_alive_reward = RewTerm(
    #     func=mdp.is_alive, weight=+1e-1  # Dense Reward of [-1e-3, 0.0] --> Max Episode Penalty: -???
    # )


class CrowdNavigationTeacherDynSFMEnvCfg(CrowdNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # add teacher lidar:
        self.scene.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=1 / self.fz_planner,  # 0.1, lidar hz =  planner hz = 10
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(0, 360),
                horizontal_res=1,
            ),
            max_distance=100.0,
            drift_range=(-0.0, 0.0),
            debug_vis=True,
            history_length=1,
            visualizer_cfg=MY_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
            mesh_prim_paths=["/World/ground", "/World/envs/env_.*/Obstacle"], # "{ENV_REGEX_NS}/Obstacle",self.scene.obstacle.prim_path
            track_mesh_transforms=True,
            # max_meshes=32,
            # mesh_ids_to_keep=[0],  # terrain id
        )

        # change terrain:
        # self.scene.terrain: AssetBaseCfg = AssetBaseCfg(
        #     prim_path="/World/ground",
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        #     spawn=GroundPlaneCfg(),
        # )
        # remove obstacles:
        # self.scene.obstacle = EMPTY_OBS_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle")

        # add observation group:
        self.observations.policy = TeacherPolicyObsCfg()

        # change terminations
        self.terminations: TeacherTerminationsCfg = TeacherTerminationsCfg()

        # change max duration
        self.episode_length_s = 120

        # change env spacing as curriculum
        self.scene.env_spacing = 8.0
        # self.curriculum: TeacherCurriculumCfg = TeacherCurriculumCfg()

        # change rewards
        self.rewards: TeacherRewardsCfg = TeacherRewardsCfg()
