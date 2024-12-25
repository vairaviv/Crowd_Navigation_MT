from omni.isaac.lab.utils import configclass

# from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
# from crowd_navigation_mt.env_config.crowd_navigation_env_cfg import CrowdNavigationEnvCfg

# from crowd_navigation_mt.env_config.crowd_navigation_env_cfg_legacy import (
#     CrowdNavigationEnvCfg,
# )

from crowd_navigation_mt.env_config.crowd_navigation_teacher_env_cfg import (
    CrowdNavigationTeacherEnvCfg,
)

from crowd_navigation_mt.env_config.crowd_navigation_teacher_dyn_env_cfg import (
    CrowdNavigationTeacherDynEnvCfg,
)

from crowd_navigation_mt.env_config.crowd_navigation_flat_env_cfg import CrowdNavigationFlatEnvCfg

from crowd_navigation_mt.env_config.crowd_navigation_SFM_base_env_cfg import SFMBaseEnvCfg

from crowd_navigation_mt.env_config.crowd_navigation_SFM_curr_env import SFMCurrEnvCfg
# from crowd_navigation_mt.env_config.crowd_navigation_recording_env_cfg import (
#     CrowdNavigationRecordingEnvCfg,
# )

# # eval envs
# from crowd_navigation_mt.env_config.stat_obs_eval_cfg import (
#     CrowdNavigationTeacherStatObsEvalEnvCfg,
# )


from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip

from crowd_navigation_mt.assets import ANYMAL_D_EXT_BASE_CFG

# ISAAC_GYM_JOINT_NAMES are already defined in the base class, they are probably also robot specific...


# @configclass
# class AnymalDNavigationEnvCfg(CrowdNavigationEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # switch robot to anymal-d
#         self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#         self.scene.robot.init_state.joint_pos = {
#             "LF_HAA": -0.13859,
#             "LH_HAA": -0.13859,
#             "RF_HAA": 0.13859,
#             "RH_HAA": 0.13859,
#             ".*F_HFE": 0.480936,  # both front HFE
#             ".*H_HFE": -0.480936,  # both hind HFE
#             ".*F_KFE": -0.761428,
#             ".*H_KFE": 0.761428,
#         }
#         self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]


@configclass
class AnymalDCrowdNavigationTeacherFlatEnvCfg(CrowdNavigationFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]


@configclass
class AnymalDCrowdNavigationTeacherEnvCfg(CrowdNavigationTeacherEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]


@configclass
class AnymalDCrowdNavigationTeacherDynEnvCfg(CrowdNavigationTeacherDynEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]


@configclass
class AnymalDCrowdNavigationTeacherDynEnvHistoryCfg(CrowdNavigationTeacherDynEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]

        #change history length
        self.observations.policy.lidar_distances_history.history_length = 10



@configclass
class AnymalDSFMCfg(SFMBaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]

        #change history length
        # self.observations.policy.lidar_distances_history.history_length = 10

        

@configclass
class AnymalDExtBaseSFMCfg(SFMBaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_EXT_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]

        #change history length
        # self.observations.policy.lidar_distances_history.history_length = 10

@configclass
class AnymalDExtSFMCurrCfg(SFMCurrEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_EXT_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]

        #change history length
        # self.observations.policy.lidar_distances_history.history_length = 10


# @configclass
# class AnymalDCrowdNavigationRecordingEnvCfg(CrowdNavigationRecordingEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # switch robot to anymal-d
#         self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#         self.scene.robot.init_state.joint_pos = {
#             "LF_HAA": -0.13859,
#             "LH_HAA": -0.13859,
#             "RF_HAA": 0.13859,
#             "RH_HAA": 0.13859,
#             ".*F_HFE": 0.480936,  # both front HFE
#             ".*H_HFE": -0.480936,  # both hind HFE
#             ".*F_KFE": -0.761428,
#             ".*H_KFE": 0.761428,
#         }
#         self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]


# """ EVALUATION ENVIRONMENTS"""


# @configclass
# class AnymalDCrowdNavigationStatObsEvalEnvCfg(CrowdNavigationTeacherStatObsEvalEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # switch robot to anymal-d
#         self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#         self.scene.robot.init_state.joint_pos = {
#             "LF_HAA": -0.13859,
#             "LH_HAA": -0.13859,
#             "RF_HAA": 0.13859,
#             "RH_HAA": 0.13859,
#             ".*F_HFE": 0.480936,  # both front HFE
#             ".*H_HFE": -0.480936,  # both hind HFE
#             ".*F_KFE": -0.761428,
#             ".*H_KFE": 0.761428,
#         }
#         self.scene.robot.spawn.scale = [1.15, 1.15, 1.15]
