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


from .crowd_navigation_stat_obs_base_env_cfg import CrowdNavigationEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from typing import Literal

from crowd_navigation_mt.terrains.config.rough import TEST_TERRAIN_CFG  # isort: skip

##
# Pre-defined configs
##
# from crowd_navigation_mt.assets.simple_obstacles import (
#     OBS_CFG,
#     EMPTY_OBS_CFG,
#     CYLINDER_HUMANOID_CFG,
#     MY_RAY_CASTER_MARKER_CFG,
#     WALL_CFG,
# )


# observation group:
# ie default lidar dist
LIDAR_HISTORY = mdp.LidarHistory(history_length=1, sensor_cfg=SceneEntityCfg("lidar"), return_pose_history=True)


@configclass
class TeacherPolicyObsCfg(ObsGroup):
    """Observations for the teacher policy"""

    # commands
    target_position = ObsTerm(
        func=mdp.generated_commands_reshaped, params={"command_name": "robot_goal", "flatten": True}  # robot_goal
    )

    # add cpg state to the proprioception group

    cpg_state = ObsTerm(func=mdp.cgp_state)

    # privileged sensor observations
    # lidar_distances = ObsTerm(
    #     func=mdp.lidar_obs_dist,
    #     params={"sensor_cfg": SceneEntityCfg("lidar"), "flatten": True},
    #     # noise=Unoise(n_min=-0.1, n_max=0.1),
    #     clip=(-100.0, 100.0),
    # )

    lidar_distances_history = ObsTerm(
        func=lambda env: LIDAR_HISTORY.get_history(env),
        clip=(-100.0, 100.0),
    )

    # lidar_mesh_velocities = ObsTerm(
    #     func=mdp.lidar_obs_vel_rel_2d,
    #     params={"sensor_cfg": SceneEntityCfg("lidar"), "flatten": True},
    #     clip=(-100.0, 100.0),
    # )

    # # optional segmentation
    # segmentation = ObsTerm(
    #     func=mdp.lidar_panoptic_segmentation,
    #     params={"sensor_cfg": SceneEntityCfg("lidar"), "flatten": True},
    # )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


eval_tracker = mdp.EvalBuffer()


@configclass
class EvalLogging(ObsGroup):
    """Observations for evaluation data logging."""

    # Episode Signals
    timeout_signal = ObsTerm(func=mdp.metrics_timeout_signal)
    termination_signal = ObsTerm(func=mdp.metrics_termination_signal)
    dones_signal = ObsTerm(func=mdp.metrics_dones_signal)

    undesired_contacts = ObsTerm(
        func=mdp.metrics_undesired_contacts,
        params={"threshold": 0.01, "body_names": [".*THIGH", ".*HIP", ".*SHANK", "base"]},
    )
    episode_length = ObsTerm(func=mdp.metrics_episode_length)

    eval_tracker = ObsTerm(func=eval_tracker.update)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP."""

    # -- tasks
    goal_reached = RewTerm(
        func=mdp.goal_reached,  # reward is high to compensate for reset penalty
        weight=1000.0,  # Sparse Reward of {0.0,0.2} --> Max Episode Reward: 2.0
        params={"distance_threshold": 0.5, "speed_threshold": 2.5},
    )

    goal_progress = RewTerm(
        func=mdp.goal_progress,
        weight=1,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    goal_closeness = RewTerm(
        func=mdp.goal_closeness,
        weight=1,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    # stage_cleared = RewTerm(
    #     func=mdp.stage_cleared,
    #     weight=50,  # Sparse Reward
    # )

    # near_goal_stability = RewTerm(
    #     func=mdp.near_goal_stability,
    #     weight=1.0,  # Dense Reward of [0.0, 0.1] --> Max Episode Reward: 1.0
    #     params={"threshold": 0.5},
    # )

    # -- penalties

    lateral_movement = RewTerm(
        func=mdp.lateral_movement,
        weight=-0.05,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )

    backward_movement = RewTerm(
        func=mdp.backwards_movement,
        weight=-0.05,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )

    episode_termination = RewTerm(
        func=mdp.is_terminated,
        weight=-500.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, weight=-0.1  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )

    no_robot_movement = RewTerm(
        func=mdp.no_robot_movement_2d,
        weight=-5,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    )

    #  penalty for being close to the obstacles
    close_to_obstacle = RewTerm(
        func=mdp.obstacle_distance,
        weight=-2.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 0.75, "dist_std": 0.2, "dist_sensor": SceneEntityCfg("lidar")},
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

    # penalty for colliding with obstacles
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", ".*HIP", ".*SHANK", "base"]),
            "threshold": 0.5,
        },
    )

    # time penalty, should encourage the agent to reach the goal as fast as possible
    is_alive_penalty = RewTerm(
        func=mdp.is_alive, weight=-1e-2  # Dense Reward of [-1e-3, 0.0] --> Max Episode Penalty: -???
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # robot_direction = mdp.DirectionCommandCfg(
    #     resampling_time_range=(100000.0, 100000.0),
    #     asset_name="robot",
    #     direction=(1.0, 0.0),
    # )

    # target pos for the robot. the agent has to learn to move to this position
    robot_goal = mdp.RobotGoalCommandCfg(
        # TODO goal should be in a spawn position next to the robots spawn position --> ensure that goal is not in an obstacle
        asset_name="robot",
        resampling_time_range=(100000.0, 100000.0),  # resample only on reset
        debug_vis=True,
        radius=5.0,
        terrain_analysis=mdp.TerrainAnalysisCfg(
            raycaster_sensor="lidar",
        ),
        # angles=[0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
        use_grid_spacing=True,
        deterministic_goal=True,
        deterministic_goal_distance_x=2,
        deterministic_goal_distance_y=2,
    )


@configclass
class TeacherTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )yy
    # thigh_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )

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
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    # eye: tuple[float, float, float] = (-60.0, 0.5, 70.0)
    eye: tuple[float, float, float] = (0.0, 0.0, 130.0)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    # lookat: tuple[float, float, float] = (-60.0, 0.0, -10000.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: Literal["world", "env", "asset_root"] = "world"
    """
    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    """
    env_index: int = 0
    asset_name: str | None = None  # "robot"


class CrowdNavigationTeacherStatObsEvalEnvCfg(CrowdNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.viewer: ViewerCfg = ViewerCfg()
        self.scene.terrain.terrain_generator = TEST_TERRAIN_CFG

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
            debug_vis=False,
            history_length=1,
            # mesh_prim_paths=["/World/ground", self.scene.obstacle.prim_path],
            mesh_prim_paths=["/World/ground"],
            track_mesh_transforms=True,
            max_meshes=1,
            mesh_ids_to_keep=[0],  # terrain id
        )

        # change terrain:
        self.scene.terrain.max_init_terrain_level = int(
            self.scene.terrain.terrain_generator.num_cols - self.commands.robot_goal.deterministic_goal_distance_x
        )
        # self.scene.terrain: AssetBaseCfg = AssetBaseCfg(
        #     prim_path="/World/ground",
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        #     spawn=GroundPlaneCfg(),
        # )

        # add observation group:
        self.observations.policy = TeacherPolicyObsCfg()
        self.observations.evaluation = EvalLogging()

        # change max duration
        self.episode_length_s = 90

        # change env spacing as curriculum
        # self.scene.env_spacing = 3.0
        # self.curriculum: TeacherCurriculumCfg = TeacherCurriculumCfg()

        # change terminations
        self.terminations: TeacherTerminationsCfg = TeacherTerminationsCfg()

        # change rewards
        self.rewards: TeacherRewardsCfg = TeacherRewardsCfg()

        # change commands
        self.commands: CommandsCfg = CommandsCfg()

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # disable curriculum for terrain generator
        self.scene.terrain.terrain_generator.curriculum = False
