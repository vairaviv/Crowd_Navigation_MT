# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from crowd_navigation_mt.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticBetaCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticBetaCompressCfg,
    RslRlPpoActorCriticBetaCompressTemporalCfg,
    RslRlPpoActorCriticBetaLidarCNNCfg,
    RslRlPpoActorCriticBetaLidar2DCNNCfg,
    RslRlPpoActorCriticBeta2DCNNSemanticCfg
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.modules import (
        ActorCriticBetaCompress,
        ActorCriticBetaRecurrentLidar,
    )


######################################################################
# PPO - SFM configs
######################################################################

SFM_TARGET_DIM = 2 
SFM_CPG_DIM = 8
SFM_LIDAR_DIM = 360
SFM_HISTORY_LENGTH_LIDAR = 10
SFM_LIDAR_EXTRA_DIM = 3

@configclass
class PPOBaseBetaSFMLidarCNNCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaLidarCNNCfg = RslRlPpoActorCriticBetaLidarCNNCfg(
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
        target_dim=SFM_TARGET_DIM,  # Target dimension of goal position,
        cpg_dim=SFM_CPG_DIM, #cpg state 
        lidar_dim=SFM_LIDAR_DIM,  
        lidar_extra_dim=SFM_LIDAR_EXTRA_DIM * SFM_HISTORY_LENGTH_LIDAR,
        lidar_history_dim=SFM_HISTORY_LENGTH_LIDAR,  
        target_cpg_layer_dim=[64],
        lidar_cnn_layer_dim=[8, 16, 16],
        lidar_cnn_kernel_sizes=[5, 5, 5],
        lidar_cnn_strides=[2, 2, 2],
        lidar_cnn_to_mlp_layer_dim=[512, 256],
        lidar_extra_mlp_layer_dim=[16],
        lidar_merge_mlp_layer_dim=[256],
        out_layer_dim=[256, 256, 128],  # navigation network
        activation="elu",
        permute_obs=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPOBaseBetaSFMLidar2DCNNCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaLidar2DCNNCfg = RslRlPpoActorCriticBetaLidar2DCNNCfg(
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
        target_dim=SFM_TARGET_DIM,  # Target dimension of goal position,
        cpg_dim=SFM_CPG_DIM, #cpg state 
        lidar_dim=SFM_LIDAR_DIM,  
        lidar_extra_dim=SFM_LIDAR_EXTRA_DIM * SFM_HISTORY_LENGTH_LIDAR,
        lidar_history_dim=SFM_HISTORY_LENGTH_LIDAR,  
        target_cpg_layer_dim=[64],
        lidar_cnn_channel_dim=[8, 16, 16],
        lidar_cnn_kernel_sizes=[5, 5, 5],
        lidar_cnn_strides=[2, 2, 2],
        lidar_cnn_to_mlp_layer_dim=[512, 256],
        lidar_extra_mlp_layer_dim=[16],
        lidar_merge_mlp_layer_dim=[256],
        out_layer_dim=[256, 256, 128],  # navigation network
        activation="elu",
        
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


##########################################
# With semantics observations
##########################################
SEM_MAP_H = int(10 / 0.2)  # observation range divided by the map resolution
SEM_MAP_W = int(10 / 0.2)
SEM_CHANNELS = 6
PROP_DIM = 2 + 8 + 3 + 3  # target_pos, cpg_state, base_lin_vel, base_ang_vel

@configclass
class PPOBaseBeta2DCNNSemanticCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBeta2DCNNSemanticCfg = RslRlPpoActorCriticBeta2DCNNSemanticCfg(
        beta_initial_logit=0.5,
        beta_initial_scale=5.0, 
        proprio_dim=PROP_DIM,
        semantic_map_dim=[SEM_CHANNELS, SEM_MAP_H, SEM_MAP_W],
        proprio_layer_dim=[64],
        semantic_cnn_channel_dim=[8, 16, 16],
        semantic_cnn_kernel_sizes=[5, 5, 5],
        semantic_cnn_strides=[2, 2, 2],
        semantic_cnn_to_mlp_layer_dim=[512, 256],
        nav_layer_dim=[256, 256, 128],  # navigation network
        activation="elu",
        
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )