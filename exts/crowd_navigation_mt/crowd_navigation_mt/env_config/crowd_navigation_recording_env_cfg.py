# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
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

import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
import os


from .crowd_navigation_dyn_obs_base_env_cfg import CrowdNavigationEnvCfg, DynObsSceneCfg, ObservationsCfg

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


# lidar observation group:
@configclass
class ObservationsRecordCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the a basic goal tracking policy"""

        target_position = ObsTerm(func=mdp.generated_commands_reshaped, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LidarObs(ObsGroup):
        """Observations to be recorded for encoder training"""

        lidar_full = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LidarLabelsObs(ObsGroup):
        lidar_full = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar_label")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
        )

        lidar_label_distances = ObsTerm(
            func=mdp.lidar_obs_dist,
            params={"sensor_cfg": SceneEntityCfg("lidar_label")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
        )

        lidar_label_mesh_velocities = ObsTerm(
            func=mdp.lidar_obs_vel_rel_2d,
            params={"sensor_cfg": SceneEntityCfg("lidar_label")},
            clip=(-100.0, 100.0),
        )

        segmentation = ObsTerm(
            func=mdp.lidar_panoptic_segmentation,
            params={"sensor_cfg": SceneEntityCfg("lidar_label")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class MeshPositions(ObsGroup):
        positions = ObsTerm(
            func=mdp.lidar_privileged_mesh_pos_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar_label")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    lidar_raw: LidarObs = LidarObs()
    lidar_2d: LidarLabelsObs = LidarLabelsObs()

    positions: MeshPositions = MeshPositions()

    policy: PolicyCfg = PolicyCfg()


# add second lidar to scene config
@configclass
class RecordingScene(DynObsSceneCfg):
    lidar_label = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.1,
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
        visualizer_cfg=MY_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
        history_length=1,
        mesh_prim_paths=["/World/ground"],
        track_mesh_transforms=True,
    )

    def __post_init__(self):
        self.lidar_label.mesh_prim_paths.append(self.obstacle.prim_path)


class CrowdNavigationRecordingEnvCfg(CrowdNavigationEnvCfg):

    scene: RecordingScene = RecordingScene(num_envs=2, env_spacing=5)
    observations: ObservationsRecordCfg = ObservationsRecordCfg()

    def __post_init__(self):
        super().__post_init__()

        # add lidar:
        self.scene.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.1,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=16,
                vertical_fov_range=(-30.0, 30.0),
                horizontal_fov_range=(0, 360),
                horizontal_res=0.2,
            ),
            max_distance=100.0,
            drift_range=(-10.0, 10.0),
            debug_vis=False,
            visualizer_cfg=MY_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
            history_length=1,
            mesh_prim_paths=["/World/ground", self.scene.obstacle.prim_path],
            track_mesh_transforms=True,
        )

        self.commands.robot_goal.debug_vis = False
