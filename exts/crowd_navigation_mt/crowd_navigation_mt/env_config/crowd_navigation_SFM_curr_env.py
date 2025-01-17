# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, RayCasterCameraCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from crowd_navigation_mt.env_config.crowd_navigation_SFM_base_env_cfg import SFMBaseEnvCfg

import crowd_navigation_mt.mdp as mdp 
# import crowd_navigation_mt.sensors import patterns
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from crowd_navigation_mt import CROWDNAV_DATA_DIR
from crowd_navigation_mt.mdp import ObservationHistoryTermCfg

from nav_tasks.mdp.commands import GoalCommandCfg, ConsecutiveGoalCommandCfg
from crowd_navigation_mt.mdp.commands import LvlConsecutiveGoalCommandCfg
from nav_collectors.collectors import TrajectorySamplingCfg
from nav_collectors.terrain_analysis import TerrainAnalysisCfg


##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from crowd_navigation_mt.assets.simple_obstacles import (
    OBS_CFG,
    EMPTY_OBS_CFG,
    CYLINDER_HUMANOID_CFG,
    MY_RAY_CASTER_MARKER_CFG,
    WALL_CFG,
)
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from crowd_navigation_mt.terrains.config import (
    ROUGH_TERRAINS_CFG,
    OBS_TERRAINS_CFG,
    OBS_TERRAINS_DYNOBS_CFG,
    DYN_CURR_OBS_TERRAIN_CFG,
)  # isort: skip
from crowd_navigation_mt.terrains.importer import SFMObstacleImporterCfg

from crowd_navigation_mt.mdp import LidarHistoryTermCfg

from crowd_navigation_mt.assets import SFMObstacleCfg

"""To improve: we have 3 separate goal_reached functions, one for logging, one for reward and one for termination

ATM all functions are getting the same threshold"""

DISTANCE_THRESHOLD = 0.5
SPEED_THRESHOLD = 2.5


ISAAC_GYM_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]

##
# Terrain definition
##

# Terrain configs
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=DYN_CURR_OBS_TERRAIN_CFG,  # OBS_TERRAINS_CFG,
    max_init_terrain_level=0,
    max_init_terrain_type=0,
    collision_group=-1,
    # env_spacing=8.0,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)

# Obstacle config
sfm_obstacle : AssetBaseCfg = SFMObstacleCfg(
    prim_path="{ENV_REGEX_NS}/SFM_Obstacle",
    spawn=sim_utils.CylinderCfg(
        radius=0.35,
        height=2,
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.005),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=None,  # sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.9, 0.6)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 5.0, 3.0)),
    collision_group=-1,
    num_levels=4,
    num_types=4,
    num_sfm_agent_increase=5,
)

##
# MDP settings
##


@configclass
class CurrCommandsCfg:
    """Command specifications for the MDP.
    In this task, the commands probably will be the goal positions for the robot to reach."""

    ##
    # Dynamic obstacle's Commands
    ##

    # target pos for the obstacle consecutive goal samplings depending on the level
    sfm_obstacle_target_pos = LvlConsecutiveGoalCommandCfg(
        asset_name="sfm_obstacle",
        resampling_time_range=(10000000.0, 10000000.0), 
        debug_vis=False,
        terrain_analysis=TerrainAnalysisCfg(
            raycaster_sensor="sfm_obstacle_lidar",
            semantic_cost_mapping=None,
        ),
        resample_distance_threshold=0.5,
    )  
   
    
    # target position for the robot, random goal sampling from the previous project, terrain Analysis doesn't work
    robot_goal = mdp.RobotLvlGoalCommandCfg(
        # TODO goal should be in a spawn position next to the robots spawn position --> ensure that goal is not in an obstacle
        asset_name="robot",
        resampling_time_range=(100000.0, 100000.0),  # resample only on reset
        debug_vis=False,
        radius=1.0,
        # terrain_analysis=mdp.TerrainAnalysisCfg(
        #     raycaster_sensor="lidar",
        # ),  # not required for generated terrains, but for moving environments
        terrain_analysis=TerrainAnalysisCfg(
            raycaster_sensor="lidar",
            semantic_cost_mapping=None,
        ),
        # angles=[0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
        use_grid_spacing=False,
    )


@configclass
class TeacherPolicyObsCfg(ObsGroup):
    """Observations for the teacher policy"""

    # commands
    target_position = ObsTerm(
        func=mdp.generated_commands_reshaped, params={"command_name": "robot_goal", "flatten": True}
    )

    cpg_state = ObsTerm(func=mdp.cgp_state)

    lidar_distances_history = LidarHistoryTermCfg(
        func=mdp.LidarHistory,
        params={"method": "get_history"},
        history_length=10,
        history_time_span=5,
        # decimation=1,
        sensor_cfg=SceneEntityCfg("lidar"),
        return_pose_history=True,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CurrEventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    # reset obstacle spawn position according to obstacle command
    # TODO: TerrainAnalysisRootReset is not of type ManagerTerm, check what the issue is
    # reset_sfm_obstacle = EventTerm(
    #     func=mdp.TerrainAnalysisRootReset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("sfm_obstacle"),
    #         "yaw_range": (-3.14, 3.14),
    #         "velocity_range": (0.0, 0.0),
    #     }
    # )
    # -----------------------------------------
    # reset only works with Command of type :class:`nav_tasks.mdp.GoalCommand` with certain attribute 
    reset_sfm_obstacle = EventTerm(
        func=mdp.reset_robot_position,
        mode="reset",
        params={
            "goal_command_generator_name": "sfm_obstacle_target_pos",
            "asset_cfg": SceneEntityCfg("sfm_obstacle"),
            "yaw_range": (-3.14, 3.14),
        }
    )
    # -----------------------------------------

    # TODO curriculum spawning
    reset_base = EventTerm(
        func=mdp.reset_robot_position_plr,
        mode="reset",
        params={
            # "additive_heading_range": {"yaw": (-1.0, +1.0)},
            "additive_heading_range": {"yaw": (-3.14, 3.14)},
            "command_name": "robot_goal",
        },
    )
    # reset_base = EventTerm(
    #     func=mdp.reset_robot_position,
    #     mode="reset",
    #     params={
    #         # "additive_heading_range": {"yaw": (-1.0, +1.0)},
    #         "yaw_range": (-3.14, 3.14),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "goal_command_generator_name": "robot_goal",
    #     },
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP."""

    # -- tasks
    goal_reached = RewTerm(
        func=mdp.goal_reached,  # reward is high to compensate for reset penalty
        weight=50.0,  # Sparse Reward of {0.0,0.2} --> Max Episode Reward: 2.0
        params={"distance_threshold": DISTANCE_THRESHOLD, "speed_threshold": SPEED_THRESHOLD},
    )

    goal_progress = RewTerm(
        func=mdp.goal_progress,
        weight=2.0,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    goal_closeness = RewTerm(
        func=mdp.goal_closeness,
        weight=1.0,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
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
    #     weight=-200.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    # )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.1  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )

    # took out because Observation can not handle history yet
    # no_robot_movement = RewTerm(
    #     func=mdp.no_robot_movement_2d,
    #     weight=-5,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    # )
    # 
    # no_robot_movement = RewTerm(
    #     func=mdp.no_robot_movement,
    #     weight=-0.1,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    #     params={"goal_distance_thresh": 0.5},
    # )

    #  penalty for being close to the obstacles
    close_to_obstacle = RewTerm(
        func=mdp.obstacle_distance,
        weight=-2.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 6.0, "dist_std": 0.5, "dist_sensor": SceneEntityCfg("lidar")},
    )

    # TODO add penality for obstacles being in front of the robot

    obstacle_in_front_narrow = RewTerm(
        func=mdp.obstacle_distance_in_front,
        weight=-1.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 5.0, "dist_std": 1.5, "dist_sensor": SceneEntityCfg("lidar"), "degrees": 60.0},
    )

    obstacle_in_front_wide = RewTerm(
        func=mdp.obstacle_distance_in_front,
        weight=-1.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 3.0, "dist_std": 0.5, "dist_sensor": SceneEntityCfg("lidar"), "degrees": 180.0},
    )

    # far_from_obstacle = RewTerm(
    #     func=mdp.obstacle_distance,
    #     weight=-0.1,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    #     params={"threshold": 1, "dist_std": 5, "dist_sensor": SceneEntityCfg("lidar")},
    # )
   

    # # penalty for colliding with obstacles
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-200.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", ".*HIP", ".*SHANK", "base"]),
    #         "threshold": 0.5,
    #     },
    # )

    # # reward for being alive
    # is_alive_reward = RewTerm(
    #     func=mdp.is_alive, weight=+1e-1  # Dense Reward of [-1e-3, 0.0] --> Max Episode Penalty: -???
    # )

    # # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)



##################################################################
# took out curriculum for now
##################################################################

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    goal_distance = CurrTerm(
        func=mdp.modify_goal_distance,
        params={
            "step_size": 0.5,
            "required_successes": 5,
        },
    )


##
# Environment configuration
##


@configclass
class SFMCurrEnvCfg(SFMBaseEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.scene.terrain = terrain
        self.scene.sfm_obstacle = sfm_obstacle
        # Consecutive Goal commands based on terrain level 
        self.commands.sfm_obstacle_target_pos = CurrCommandsCfg().sfm_obstacle_target_pos
        # self.commands = CurrCommandsCfg()
        rewards : TeacherRewardsCfg = TeacherRewardsCfg()
        events: CurrEventCfg = CurrEventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

