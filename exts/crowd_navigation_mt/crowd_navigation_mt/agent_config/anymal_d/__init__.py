# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, navigation_env_cfg


""" Navigation environments """

# gym.register(
#     id="Isaac-CrowdNavigation-Flat-Anymal-D-v0",
#     entry_point="omni.isaac.orbit.envs:RLTaskEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.AnymalDNavigationEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOCfg,
#     },
# )

# flat terrain for testing
gym.register(
    id="Isaac-CrowdNavigation-Teacher-Flat-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherBaseCfg,
    },
)

# flat terrain with Actor Critic Beta Module
gym.register(
    id="Isaac-CrowdNavigation-Teacher-Flat-Beta-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOBaseBetaCfg,
    },
)

# static obstacles PPO Base
gym.register(
    id="Isaac-CrowdNavigation-Teacher-StatObs-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherBaseCfg,
    },
)

# static obstacles PPO Beta 
gym.register(
    id="Isaac-CrowdNavigation-Teacher-StatObs-Beta-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherBetaCfg,
    },
)

# static obstacles PPO CONV BetaCompressed NoGru
gym.register(
    id="Isaac-CrowdNavigation-Teacher-StatObs-Conv_NoGru-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherNoGruConvCfg,
    },
)

# dynamic obstacles PPO Beta
gym.register(
    id="Isaac-CrowdNavigation-Teacher-DynObs-Beta-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherDynEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherDynBetaCfg,
    },
)


# dynamic obstacles PPO CONV BetaCompressed NoGru
gym.register(
    id="Isaac-CrowdNavigation-Teacher-DynObs-Conv_NoGru-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherDynEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherNoGruConvCfg,
    },
)




# # staitc obstacles with heightscan
# gym.register(
#     id="Isaac-CrowdNavigation-Teacher-StatObs-Height-Anymal-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacheConvHeightCfg,
#     },
# )

# # static obstacles without gru
# gym.register(
#     id="Isaac-CrowdNavigation-Teacher-StatObs-NoGru-Anymal-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacheNoGruConvCfg,
#     },
# )


# gym.register(
#     id="Isaac-CrowdNavigation-Teacher-DynObs-Anymal-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherDynEnvCfg,  # AnymalDCrowdNavigationTeacherDynEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacheDynConvCfg,
#     },
# )


# """ EVALUATION ENVIRONMENTS """
# gym.register(
#     id="Isaac-CrowdNavigation-Teacher-StatObs-EVAL-Anymal-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationStatObsEvalEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacheConvCfg,
#     },
# )


#################################################
# Trials
#################################################

gym.register(
    id="Isaac-CrowdNavigation-Teacher-DynObs-Anymal-D-Trials",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherDynEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherBaseCfg,
    },
)

gym.register(
    id="Isaac-CrowdNavigation-Teacher-StatObs-Anymal-D-Trials",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AnymalDCrowdNavigationTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPOTeacherBaseCfg,
    },
)