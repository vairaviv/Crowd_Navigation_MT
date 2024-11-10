# Copyright (c) 2022-2024, The ORBIT Project Developers.
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
    RslRlPpoActorCriticBetaLidarTemporalCfg,
    RslRlPpoActorCriticBetaRecurrentLidarCfg,
    RslRlPpoActorCriticBetaRecurrentLidarCnnCfg,
    RslRlPpoActorCriticBetaRecurrentLidarHeightCnnCfg,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.modules import (
        ActorCriticBetaCompress,
        ActorCriticBetaRecurrentLidar,
    )


# basic compress mlp
@configclass
class PPOBaseBetaCompressCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 1234
    policy: RslRlPpoActorCriticBetaCompressCfg = RslRlPpoActorCriticBetaCompressCfg(
        input_dims=[2 + 8, 180],  # target pos, 2d lidar obs. Sum needs to equal observation dimensions
        actor_hidden_dims_per_split=[[128], [256, 128, 64]],
        critic_hidden_dims_per_split=[
            [128],
            [256, 128, 64],
        ],
        # add proproiception, velicoty and / or history of distances
        actor_out_hidden_dims=[256, 256, 128],
        critic_out_hidden_dims=[256, 256, 128],
        activation="elu",
        module_types=["mlp", "mlp"],
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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
class PPOBaseCfg(RslRlPpoActorCriticCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 1234
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        # input_dims=[2 + 8, 360],  # target pos, 2d lidar obs. Sum needs to equal observation dimensions
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        # add proproiception, velicoty and / or history of distances
        # actor_out_hidden_dims=[256, 256, 128],
        # critic_out_hidden_dims=[256, 256, 128],
        activation="elu",
        # module_types=["mlp", "mlp"],
        # beta_initial_logit=0.5,
        # beta_initial_scale=5.0,
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

NON_LIDAR_DIM = 2 + 8  # commanded pos, proprioception
HISTORY_LENGTH_STAT = 1
HISTORY_LENGTH_DYN = 10
DIM_LIDAR = 360
LIDAR_EXTRA_DIM = 3  # history length 5
# 875631


# compress with history
@configclass
class PPOBaseBetaCompressPreTemporalCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaCompressTemporalCfg = RslRlPpoActorCriticBetaCompressTemporalCfg(
        input_dims=[2 + 8, DIM_LIDAR * HISTORY_LENGTH_STAT],
        single_dims=[2 + 8, DIM_LIDAR],
        n_parallels=[1, HISTORY_LENGTH_STAT],
        actor_siamese_hidden_dims_per_split=[[128], [256, 128]],
        critic_siamese_hidden_dims_per_split=[[128], [256, 128]],
        actor_out_hidden_dims_per_split=[[64], [128]],
        critic_out_hidden_dims_per_split=[[64], [128]],
        actor_out_hidden_dims=[256, 128],
        critic_out_hidden_dims=[256, 128],
        activation="elu",
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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


# compress with gru, only lidar
@configclass
class PPOBaseBetaLidarTemporalCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaRecurrentLidarCfg = RslRlPpoActorCriticBetaRecurrentLidarCfg(
        non_lidar_dim=NON_LIDAR_DIM,
        lidar_dim=DIM_LIDAR,
        # lidar_extra_dim=LIDAR_EXTRA_DIM,
        # history_length=HISTORY_LENGTH_STAT,
        non_lidar_layer_dims=[128],
        lidar_compress_layer_dims=[256, 128],
        # lidar_extra_in_dims=[16],
        # lidar_extra_merge_mlp_dims=[128, 128],
        history_processing_mlp_dims=[256, 256],
        out_layer_dims=[256, 128],
        gru_dim=256,
        gru_layers=1,
        activation="elu",
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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


# gru with lidar cnn, multi channel lidar input
@configclass
class PPOBaseBetaLidarConvCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaRecurrentLidarCnnCfg = RslRlPpoActorCriticBetaRecurrentLidarCnnCfg(
        non_lidar_dim=NON_LIDAR_DIM,
        lidar_dim=DIM_LIDAR,
        num_lidar_channels=1,
        lidar_extra_dim=LIDAR_EXTRA_DIM,
        non_lidar_layer_dims=[64],
        lidar_compress_conv_layer_dims=[8, 16, 16],
        lidar_compress_conv_kernel_sizes=[5, 5, 5],
        lidar_compress_conv_strides=[2, 2, 2],
        lidar_compress_conv_to_mlp_dims=[512, 256],
        lidar_extra_in_dims=[16],
        lidar_merge_mlp_dims=[256],
        history_processing_mlp_dims=[256],
        out_layer_dims=[256, 256, 128],  # planning net
        gru_dim=512,
        gru_layers=1,
        activation="elu",
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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
class PPOBaseBetaLidarHeighScanConvCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaRecurrentLidarHeightCnnCfg = RslRlPpoActorCriticBetaRecurrentLidarHeightCnnCfg(
        non_lidar_dim=NON_LIDAR_DIM,
        lidar_dim=DIM_LIDAR,
        num_lidar_channels=1,
        lidar_extra_dim=LIDAR_EXTRA_DIM,
        heigh_scan_dims=(41, 41, 1),
        non_lidar_layer_dims=[64],
        lidar_compress_conv_layer_dims=[8, 8, 16],
        lidar_compress_conv_kernel_sizes=[7, 5, 5],
        lidar_compress_conv_strides=[2, 2, 2],
        lidar_compress_conv_to_mlp_dims=[512, 256],
        lidar_extra_in_dims=[32],
        height_conv_channels=[8, 16, 16],
        height_conv_kernel_sizes=[5, 5, 5],
        height_conv_strides=[2, 2, 2],
        heigh_conv_max_pool_kernel_sizes=[1, 1, 1],
        height_conv_to_mlp_dims=[265],
        lidar_merge_mlp_dims=[256],
        history_processing_mlp_dims=[256],
        out_layer_dims=[256, 256, 128],  # planning net
        gru_dim=256,
        gru_layers=1,
        activation="elu",
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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


# base beta
@configclass
class PPOBaseBetaCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    policy = RslRlPpoActorCriticBetaCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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


######################################################################
# PPO - Specific Configuration
######################################################################
class PPOTeacherBaseCfg(PPOBaseCfg):
    run_name = "PPO_Base"
    device = "cuda"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000

@configclass
class PPOCfg(PPOBaseBetaCfg):
    run_name = "PPO_Beta"

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [64, 128, 128]
        self.policy.critic_hidden_dims = [64, 128, 128]


@configclass
class PPOTeacherCfg(PPOBaseBetaCompressCfg):
    run_name = "PPO_BetaTeacher"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000


@configclass
class PPOTeacherTempCfg(PPOBaseBetaCompressPreTemporalCfg):
    run_name = "PPO_BetaTeacher"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 30000


@configclass
class PPOTeacherPreTempCfg(PPOBaseBetaCompressPreTemporalCfg):
    run_name = "PPO_BetaTeacher"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 30000


@configclass
class PPOTeacherLidarCfg(PPOBaseBetaLidarTemporalCfg):
    run_name = "PPO_BetaTeacher"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 30000
        # self.policy.history_length = HISTORY_LENGTH_STAT


@configclass
class PPOTeacherLidarDynCfg(PPOBaseBetaLidarTemporalCfg):
    run_name = "PPO_BetaTeacher"
    strict_loading = False

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 30000
        self.policy.lidar_dim = DIM_LIDAR * 3
        self.policy.lidar_compress_layer_dims = [512, 256]
        self.policy.gru_dim = 512
        self.lidar_compress_conv_to_mlp_dims = ([64],)
        lidar_extra_in_dims = ([8],)
        lidar_merge_mlp_dims = ([256],)
        history_processing_mlp_dims = ([256, 256],)
        out_layer_dims = ([256, 128],)
        # self.policy.history_length = HISTORY_LENGTH_DYN


NUM_ITERATIONS = 1_000


@configclass
class PPOTeacheConvCfg(PPOBaseBetaLidarConvCfg):
    run_name = "Static_Gru"
    wandb_project = "crowd_navigation_static_gru_test"
    wandb_name = "Static_Gru"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = NUM_ITERATIONS


@configclass
class PPOTeacheNoGruConvCfg(PPOBaseBetaLidarConvCfg):
    run_name = "Static_NoGru"
    wandb_project = "crowd_navigation_static_gru_test"
    wandb_name = "Static_NoGru"
    # strict_loading = False

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = NUM_ITERATIONS
        self.policy.gru_layers = 0


@configclass
class PPOTeacheDynConvCfg(PPOBaseBetaLidarConvCfg):
    run_name = "PPO_BetaTeacher"
    strict_loading = False
    wandb_project = "crowd_navigation"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.policy.num_lidar_channels = 3  # dist, xy vel

        # new net:
        self.policy.lidar_compress_conv_layer_dims = [2, 2, 2]
        self.policy.lidar_compress_conv_kernel_sizes = [5, 5, 5]
        self.policy.lidar_compress_conv_strides = [4, 4, 4]
        self.policy.lidar_compress_conv_to_mlp_dims = [64]
        self.policy.gru_dim = 16
        self.policy.lidar_compress_conv_to_mlp_dims = [16]
        self.policy.lidar_merge_mlp_dims = [16]
        self.policy.history_processing_mlp_dims = [16]
        self.policy.out_layer_dims = [16]


##
# with heigh scan
##


@configclass
class PPOTeacheConvHeightCfg(PPOBaseBetaLidarHeighScanConvCfg):
    run_name = "Static_Gru"
    wandb_project = "crowd_navigation_static_gru_test"
    wandb_name = "Static_Gru"
    strict_loading = False

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = NUM_ITERATIONS


@configclass
class PPOTeacheNoGruHeightConvCfg(PPOBaseBetaLidarHeighScanConvCfg):
    run_name = "Static_NoGru"
    wandb_project = "crowd_navigation_static_gru_test"
    wandb_name = "Static_NoGru"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = NUM_ITERATIONS
        self.policy.gru_layers = 0


######################################################################
# PPO - Dev Configuration
######################################################################
@configclass
class PPOCfgDEV(PPOCfg):
    logger = "tensorboard"
    run_name = "Debug"
    # max_iterations = 10



######################################################################
# PPO - CrowdNavigation Configurations PLR
######################################################################

NON_LIDAR_DIM = 2 + 8  # commanded pos, proprioception
HISTORY_LENGTH_STAT = 1
HISTORY_LENGTH_DYN = 10
DIM_LIDAR = 360
LIDAR_EXTRA_DIM = 3  # history length 5
# 875631

# gru with lidar cnn, multi channel lidar input
@configclass
class PPOBaseBetaLidarConvCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "crowd_navigation"
    empirical_normalization = False
    seed = 12345
    policy: RslRlPpoActorCriticBetaRecurrentLidarCnnCfg = RslRlPpoActorCriticBetaRecurrentLidarCnnCfg(
        non_lidar_dim=NON_LIDAR_DIM,
        lidar_dim=DIM_LIDAR,
        num_lidar_channels=1,
        lidar_extra_dim=LIDAR_EXTRA_DIM,
        non_lidar_layer_dims=[64],
        lidar_compress_conv_layer_dims=[8, 16, 16],
        lidar_compress_conv_kernel_sizes=[5, 5, 5],
        lidar_compress_conv_strides=[2, 2, 2],
        lidar_compress_conv_to_mlp_dims=[512, 256],
        lidar_extra_in_dims=[16],
        lidar_merge_mlp_dims=[256],
        history_processing_mlp_dims=[256],
        out_layer_dims=[256, 256, 128],  # planning net
        gru_dim=512,
        gru_layers=1,
        activation="elu",
        beta_initial_logit=0.5,
        beta_initial_scale=5.0,
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



NUM_ITERATIONS = 1_000


@configclass
class PPOTeacheConvCfg(PPOBaseBetaLidarConvCfg):
    run_name = "Static_Gru"
    wandb_project = "crowd_navigation_static_gru_test"
    wandb_name = "Static_Gru"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = NUM_ITERATIONS


@configclass
class PPOTeacheNoGruConvCfg(PPOBaseBetaLidarConvCfg):
    run_name = "Static_NoGru"
    wandb_project = "crowd_navigation_static_gru_test"
    wandb_name = "Static_NoGru"
    # strict_loading = False

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = NUM_ITERATIONS
        self.policy.gru_layers = 0


