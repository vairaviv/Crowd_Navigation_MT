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

import crowd_navigation_mt.mdp as mdp 
# import crowd_navigation_mt.sensors import patterns
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from crowd_navigation_mt import CROWDNAV_DATA_DIR
from crowd_navigation_mt.mdp import ObservationHistoryTermCfg

from nav_tasks.mdp.commands import GoalCommandCfg, ConsecutiveGoalCommandCfg
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
)  # isort: skip
from crowd_navigation_mt.terrains.importer import SFMObstacleImporterCfg

from crowd_navigation_mt.mdp import LidarHistoryTermCfg

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
# Scene definition
##


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=10000.0),
    )

@configclass 
class SFMObsSceneCfg(EmptySceneCfg):

    # TODO create a class to add SFM obstacles and make it initialize random num of dyn obstacles
    # sfm_obstacle: SFMObstacleImporterCfg = SFMObstacleImporterCfg()
    
    # def __post_init__(self):
    #     sfm_obstacle: SFMObstacleImporterCfg = SFMObstacleImporterCfg()
    
    # Terrain configs
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBS_TERRAINS_DYNOBS_CFG,  # OBS_TERRAINS_CFG,
        # max_init_terrain_level=2,
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
    
    ##
    # Robots configs
    ##
    
    # robots
    robot: ArticulationCfg = MISSING

    # robots sensors
    foot_scanner_lf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    foot_scanner_rf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    foot_scanner_lh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    foot_scanner_rh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    lidar : RayCasterCfg = MISSING


    ##
    # Obstacle configs
    ##

    # obstacle
    # trimesh is not working for cylinders and therefore it cant get the mesh_prim_path
    sfm_obstacle : AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SFM_Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.35,
            height=2,
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=None,  # sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.9, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 5.0, 3.0)),
        collision_group=-1,
    )

    # sfm_obstacle.replace(
    #     prim_path="{ENV_REGEX_NS}/SFM_Obstacle",
    # )

    # PLRs obstacle config, useful for debugging
    # sfm_obstacle : AssetBaseCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/SFM_Obstacle",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.75, 0.75, 2.0),  # (0.3, 0.75, 2.0)
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=None,  # sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(metallic=0.2, diffuse_color=(0.6, 0.3, 0.0)),
    #     ),
    #     collision_group=-1,
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 1), lin_vel=(1.0, 0.0, 0.0)),
    # )

    # obstacles sensors
    sfm_obstacle_lidar : RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/SFM_Obstacle",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)), # in the center of the rigid body
        attach_yaw_only=True,
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.0, 1.0]),
        pattern_cfg = patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=1,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], # TODO add robots and other obstacles here
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP.
    In this task, the commands probably will be the goal positions for the robot to reach."""

    ##
    # Dynamic obstacle's Commands
    ##

    # # target pos for the obstacle consecutive goal samplings from Nav-Suite
    sfm_obstacle_target_pos = ConsecutiveGoalCommandCfg(
        asset_name="sfm_obstacle",
        resampling_time_range=(10000000.0, 10000000.0), 
        debug_vis=True,
        terrain_analysis=TerrainAnalysisCfg(
            raycaster_sensor="sfm_obstacle_lidar",
            semantic_cost_mapping=None,
        ),
        resample_distance_threshold=0.5,
    )
    # --------------------------------------------------------
    # # target pos for the obstacle random goal samplings from Nav-Suite
    # sfm_obstacle_target_pos = GoalCommandCfg(
    #     asset_name="sfm_obstacle",
    #     resampling_time_range=(10.0, 20.0), 
    #     debug_vis=True,
    #     trajectory_config={
    #         "num_paths": [100],
    #         "max_path_length": [10.0],
    #         "min_path_length": [2.0],
    #     },
    #     z_offset_spawn=0.2,
    #     infite_sampling=True,
    #     max_trajectories=50,  # None
    #     traj_sampling=TrajectorySamplingCfg(
    #         sample_points=100,  # 1000
    #         height=1.05,
    #         enable_saved_paths_loading=True,
    #         terrain_analysis=TerrainAnalysisCfg(
    #             raycaster_sensor="sfm_obstacle_lidar",
    #             semantic_cost_mapping=None,
    #         )
    #     ),
    #     reset_pos_term_name="reset_sfm_obstacle",
    # )
    # --------------------------------------------------------
    # sfm_obstacle_target_pos = mdp.UniformPose2dCommandCfg(
    #     asset_name="sfm_obstacle",
    #     resampling_time_range=(1000000, 10000000),
    #     simple_heading=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(
    #         pos_x=(0.0, 0.0),
    #         pos_y=(0.0, 0.0),
    #     ),
    #     debug_vis=True
    # )
    # --------------------------------------------------------

    
    ##
    # Robot's Commands
    ##

    # # target position for the robot, random goal sampling from the Nav-Suite
    # robot_goal = GoalCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(100000.0, 100000.0),  # resample only on reset
    #     debug_vis=True,
    #     trajectory_config={
    #         "num_paths": [100],
    #         "max_path_length": [10.0],
    #         "min_path_length": [2.0],
    #     },
    #     z_offset_spawn=0.2,
    #     infite_sampling=True,
    #     max_trajectories=10, # None
    #     traj_sampling=TrajectorySamplingCfg(
    #         sample_points=100,  # 1000
    #         height=0.5,
    #         enable_saved_paths_loading=True,
    #         terrain_analysis=TerrainAnalysisCfg(
    #             raycaster_sensor="lidar",
    #             semantic_cost_mapping=None,
    #         )
    #     )
    # )
    
    # target position for the robot, random goal sampling from the previous project, terrain Analysis doesn't work
    robot_goal = mdp.RobotGoalCommandCfg(
        # TODO goal should be in a spawn position next to the robots spawn position --> ensure that goal is not in an obstacle
        asset_name="robot",
        resampling_time_range=(100000.0, 100000.0),  # resample only on reset
        debug_vis=True,
        radius=1.0,
        terrain_analysis=mdp.TerrainAnalysisCfg(
            raycaster_sensor="lidar",
        ),  # not required for generated terrains, but for moving environments
        # angles=[0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
        use_grid_spacing=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    ##
    # Dynamic Obstacle's action
    ##

    # Social Force Model implementation, currently position control 
    sfm_obstacle_velocity = mdp.SFMActionCfg(
        asset_name="sfm_obstacle",
        low_level_decimation=4,
        use_raw_actions=True,
        observation_group="sfm_obstacle_control_obs",
        robot_visible=False, # currently not implemented still a TODO
        robot_radius=1.5,
        debug_vis=False,
        max_sfm_velocity=0.2,
        stat_obstacle_radius=1.0,
        dyn_obstacle_radius=1.0,
        command_term_name="sfm_obstacle_target_pos",
        obstacle_sensor="sfm_obstacle_lidar",
    )

    # --------------------------------------------------------
    # PD-Controller, works like a velocity controller
    # sfm_obstacle_positions = mdp.ObstacleActionTermSimpleCfg(
    #     asset_name="sfm_obstacle",
    #     max_velocity=1,
    #     max_acceleration=5,
    #     max_rotvel=6,
    #     obstacle_center_height=1.05,
    #     raycaster_sensor="sfm_obstacle_lidar",
    # )
    # --------------------------------------------------------

    ##
    # Robot's Actions
    ##

    velocity_command = mdp.PerceptiveNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        ),
        low_level_decimation=4,
        low_level_policy_file=os.path.join(
            CROWDNAV_DATA_DIR, "Policies", "perceptive_locomotion_jit.pt"
        ),
        reorder_joint_list=ISAAC_GYM_JOINT_NAMES,
        observation_group="low_level_policy",
        scale=[1.5, 0.5, 2.0],
        offset=[-0.25, -0.25, -1.0],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # lowlevel policy
    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        # Proprioception
        wild_anymal = ObsTerm(
            func=mdp.wild_anymal,
            params={
                "action_term": "velocity_command",
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True),
            },
        )

        # Exterocpetion
        foot_scan_lf = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_lf"), "offset": 0.05},
            scale=10.0,
        )
        foot_scan_rf = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_rf"), "offset": 0.05},
            scale=10.0,
        )
        foot_scan_lh = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_lh"), "offset": 0.05},
            scale=10.0,
        )
        foot_scan_rh = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_rh"), "offset": 0.05},
            scale=10.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # obstacle positions
    @configclass
    class ObstacleControlCfg(ObsGroup):
        """Observations for obstacle group."""

        position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "sfm_obstacle_target_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DataLoggingCfg(ObsGroup):
        """Observations for data logging."""
        # Positions
        robot_position = ObsTerm(func=mdp.metrics_robot_position)
        start_position = ObsTerm(func=mdp.metrics_start_position)
        goal_position = ObsTerm(func=mdp.metrics_goal_position)
        # Path Length
        path_length = ObsTerm(func=mdp.metrics_path_length)
        # Episode Signals
        timeout_signal = ObsTerm(func=mdp.metrics_timeout_signal)
        termination_signal = ObsTerm(func=mdp.metrics_termination_signal)
        dones_signal = ObsTerm(func=mdp.metrics_dones_signal)
        goal_reached = ObsTerm(
            func=mdp.metrics_goal_reached, params={"distance_threshold": DISTANCE_THRESHOLD, "speed_threshold": SPEED_THRESHOLD}
        )  # do not care about speed
        undesired_contacts = ObsTerm(
            func=mdp.metrics_undesired_contacts,
            params={"threshold": 0.01, "body_names": [".*THIGH", ".*HIP", ".*SHANK", "base"]},
        )
        episode_length = ObsTerm(func=mdp.metrics_episode_length)

        # For Computing Rewards
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
            self.concatenate_terms = False

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

    # # observation groups
    # # observations for the low-level pretrained policy (not obs of the actual agent)
    low_level_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()

    # # obstacles
    sfm_obstacle_control_obs: ObstacleControlCfg = ObstacleControlCfg()

    # logging
    metrics: DataLoggingCfg = DataLoggingCfg()

    # policy
    policy: ObsGroup = TeacherPolicyObsCfg()


@configclass
class EventCfg:
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
    # reset_sfm_obstacle = EventTerm(
    #     func=mdp.reset_robot_position,
    #     mode="reset",
    #     params={
    #         "goal_command_generator_name": "sfm_obstacle_target_pos",
    #         "asset_cfg": SceneEntityCfg("sfm_obstacle"),
    #         "yaw_range": (-3.14, 3.14),
    #     }
    # )
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
class RewardsCfg:
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
    #     weight=-500.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
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

    #  penalty for being close to the obstacles
    close_to_obstacle = RewTerm(
        func=mdp.obstacle_distance,
        weight=-2.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 2.0, "dist_std": 0.5, "dist_sensor": SceneEntityCfg("lidar")},
    )

    # TODO add penality for obstacles being in front of the robot

    obstacle_in_front_narrow = RewTerm(
        func=mdp.obstacle_distance_in_front,
        weight=-1.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 2.0, "dist_std": 1.5, "dist_sensor": SceneEntityCfg("lidar"), "degrees": 30.0},
    )

    obstacle_in_front_wide = RewTerm(
        func=mdp.obstacle_distance_in_front,
        weight=-1.0,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"threshold": 1.0, "dist_std": 0.5, "dist_sensor": SceneEntityCfg("lidar"), "degrees": 60.0},
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

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )
    # thigh_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=["base"]  # "base", ".*THIGH", ".*HIP", ".*SHANK",
            ),
            "threshold": 1.0,
        },
    )

    goal_reached = DoneTerm(
        func=mdp.at_goal,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "distance_threshold": DISTANCE_THRESHOLD,
            "speed_threshold": SPEED_THRESHOLD,
            "goal_cmd_name": "robot_goal",
        },
        # TODO: @vairaviv check what this timeout does exactly
        time_out=True,
    )

    # update_commands = DoneTerm(
    #     func=mdp.update_command_on_termination,
    #     params={"goal_cmd_name": "robot_goal"},
    # )

##################################################################
# took out curriculum for now
##################################################################

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     goal_distance = CurrTerm(
#         func=mdp.modify_goal_distance,
#         params={
#             "step_size": 0.5,
#             "required_successes": 5,
#         },
#     )


##
# Environment configuration
##


@configclass
class SFMBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SFMObsSceneCfg = SFMObsSceneCfg(num_envs=2, env_spacing=2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    # rewards: RewardsCfg = RewardsCfg()
    rewards : RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.fz_planner = 10  # Hz
        self.episode_length_s = 120

        # DO NOT CHANGE
        self.decimation = int(200 / self.fz_planner)  # low/high level planning runs at 25Hz
        self.low_level_decimation = 4  # low level controller runs at 50Hz

        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        # self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

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
            visualizer_cfg=MY_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
            mesh_prim_paths=["/World/ground", "/World/envs/env_.*/SFM_Obstacle"],  # self.scene.sfm_obstacle.prim_path, "/World/envs/env_.*/SFM_Obstacle", "{ENV_REGEX_NS}/SFM_Obstacle", self.scene.obstacle.prim_path
            # mesh_prim_paths=["/World/ground"], # TODO add the obstacles in the lidar mesh_prim_path
            track_mesh_transforms=True,
            # max_meshes=32,
            # mesh_ids_to_keep=[0],  # terrain id
        )

