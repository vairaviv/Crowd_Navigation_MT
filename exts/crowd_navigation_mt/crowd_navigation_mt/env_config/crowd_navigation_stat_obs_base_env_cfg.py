# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg 
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

import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR as ORBIT_ASSETS_DATA_DIR
import os

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from crowd_navigation_mt.terrains.test_terrains_cfg import OBS_TERRAINS_CFG
from nav_tasks.sensors import adjust_ray_caster_camera_image_size, ZED_X_MINI_WIDE_RAYCASTER_CFG, FootScanPatternCfg

"""To improve: we have 3 separate goal_reached functions, one for logging, one for reward and one for termination"""

##
# Scene definition
##

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


OBSERVATION_HISTORY_CLASS = mdp.ObservationHistory(history_length_actions=1, history_length_positions=20)
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG, OBS_TERRAINS_CFG  # isort: skip


@configclass
class StatObsScene(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBS_TERRAINS_CFG,
        max_init_terrain_level=2,
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

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    foot_scanner_lf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    foot_scanner_rf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    foot_scanner_lh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    foot_scanner_rh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    lidar: RayCasterCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=10000.0),
    )


##
# MDP settings
##


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
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_command = mdp.PerceptiveNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        ),

        # code still from orbit
        # low_level_decimation=4,
        # low_level_policy_file=os.path.join(
        #     ORBIT_ASSETS_DATA_DIR, "Robots/RSL-ETHZ/ANYmal-D", "perceptive_locomotion_jit.pt"
        # ),
        # reorder_joint_list=ISAAC_GYM_JOINT_NAMES,
        # observation_group="low_level_policy",
        # scale=[1, 0.5, 2.0],
        # offset=[-0.25, -0.25, -1.0],
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

    @configclass
    class DataLoggingCfg(ObsGroup):
        """Observations for data logging."""

        # Positions
        robot_position = ObsTerm(func=mdp.metrics_robot_position)
        # start_position = ObsTerm(func=mdp.metrics_start_position)
        # goal_position = ObsTerm(func=mdp.metrics_goal_position)
        # Path Length
        # path_length = ObsTerm(func=mdp.metrics_path_length)
        # Episode Signals
        timeout_signal = ObsTerm(func=mdp.metrics_timeout_signal)
        termination_signal = ObsTerm(func=mdp.metrics_termination_signal)
        dones_signal = ObsTerm(func=mdp.metrics_dones_signal)
        # goal_reached = ObsTerm(
        #     func=mdp.metrics_goal_reached, params={"distance_threshold": 0.5, "speed_threshold": 1.2}
        # )  # do not care about speed
        undesired_contacts = ObsTerm(
            func=mdp.metrics_undesired_contacts,
            params={"threshold": 0.01, "body_names": [".*THIGH", ".*HIP", ".*SHANK", "base"]},
        )
        episode_length = ObsTerm(func=mdp.metrics_episode_length)

        # For Computing Rewards
        robot_position_history = ObsTerm(func=OBSERVATION_HISTORY_CLASS.get_histroy_of_positions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # # observation groups
    # # observations for the low-level pretrained policy (not obs of the actual agent)
    low_level_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()

    # logging
    metrics: DataLoggingCfg = DataLoggingCfg()

    # policy
    policy: ObsGroup = MISSING


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

    # reset
    # TODO curriculum spawning
    # reset_base = EventTerm(
    #     func=mdp.reset_robot_position,
    #     mode="reset",
    #     params={
    #         "additive_heading_range": {"yaw": (-1.0, +1.0)},
    #         # "additive_heading_range": {"yaw": (-3.14, 3.14)},
    #         "command_name": "robot_direction",
    #     },
    # )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.2, 0.2), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.5, 0.5),
            },
        },
    )

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
        weight=500.0,  # Sparse Reward of {0.0,0.2} --> Max Episode Reward: 2.0
        params={"distance_threshold": 0.5, "speed_threshold": 0.1},
    )
    goal_progress = RewTerm(
        func=mdp.goal_progress,
        weight=0.5,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )
    near_goal_stability = RewTerm(
        func=mdp.near_goal_stability,
        weight=1.0,  # Dense Reward of [0.0, 0.1] --> Max Episode Reward: 1.0
        params={"threshold": 0.5},
    )

    # -- penalties
    lateral_movement = RewTerm(
        func=mdp.lateral_movement,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    backward_movement = RewTerm(
        func=mdp.backwards_movement,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    episode_termination = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, weight=-0.1  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )

    # modification from previous project not implemented in IsaacLab and currently not needed
    # no_robot_movement = RewTerm(
    #     func=mdp.no_robot_movement,
    #     weight=-0.1,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
    #     params={"goal_distance_thresh": 0.5},
    # )

    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
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
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    goal_distance = CurrTerm(
        func=mdp.modify_goal_distance_in_steps,
        params={
            "update_rate_steps": 100 * 48,
            "min_path_length_range": (0.0, 2.0),
            "max_path_length_range": (5.0, 15.0),
            "step_range": (50 * 48, 1500 * 48),
        },
    )


##
# Environment configuration
##


@configclass
class CrowdNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: StatObsScene = StatObsScene(num_envs=2, env_spacing=5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.fz_planner = 10  # Hz
        self.episode_length_s = 30

        # DO NOT CHANGE
        self.decimation = int(200 / self.fz_planner)  # ratio of actions to physics updates
        self.low_level_decimation = 4  # low level controller runs at 50Hz

        # simulation settings
        self.sim.dt = 0.005  # 200Hz
        self.sim.disable_contact_processing = True
        # self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
