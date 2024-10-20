from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from crowd_navigation_mt.utils import RslRlPpoActorCriticBetaRecurrentLidarCnnCfg


@configclass
class PPOBaseCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "crowd_navigation_mt"
    wandb_project = "crowd_navigation_mt"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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


######################################################################
# PPO - Specific Configuration
######################################################################
@configclass
class PPOCfg(PPOBaseCfg):
    logger = "wandb"
    run_name = "PPO"


######################################################################
# PPO - Dev Configuration
######################################################################
@configclass
class PPOCfgDEV(PPOCfg):
    logger = "tensorboard"
    run_name = "Debug"



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



NUM_ITERATIONS = 10_000


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

