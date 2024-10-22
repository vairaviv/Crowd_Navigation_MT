# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from crowd_navigation_mt.env_config import (
    env_cfg_base,
    flat_dyn_base_env_cfg
) #crowd_navigation_stat_obs_base_env_cfg,

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Navigation-NavigationTemplate-PPO-Anymal-D-DEV",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavigationTemplateEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)
gym.register(
    id="Isaac-Navigation-NavigationTemplate-PPO-Anymal-D-TRAIN",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavigationTemplateEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfg",
    },
)
gym.register(
    id="Isaac-Navigation-NavigationTemplate-PPO-Anymal-D-PLAY",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavigationTemplateEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)

# gym.register(
#     id="Isaac-Navigation-CrowdNavPLR-PPO-Anymal-D-Train",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": crowd_navigation_stat_obs_base_env_cfg.CrowdNavigationEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOTeacheConvCfg",

#     }
# )

gym.register(
    id="Isaac-Navigation-SimpleDynObstNavigation-PPO-Anymal-D-DEV",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": flat_dyn_base_env_cfg.SimpleDynEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",

    }
)