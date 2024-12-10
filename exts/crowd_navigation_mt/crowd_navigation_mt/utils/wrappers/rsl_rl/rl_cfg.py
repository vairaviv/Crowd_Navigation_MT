# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticBetaCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticBeta"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticBetaCompressCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticBetaCompress"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    input_dims: list[int] = MISSING
    """The input dimensions of the sub mlps."""

    actor_hidden_dims_per_split: list[list[int]] = MISSING
    """The hidden dimensions of the actor sub mpls."""

    critic_hidden_dims_per_split: list[list[int]] = MISSING
    """The hidden dimensions of the critic sub mlps."""

    actor_out_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor output mlp."""

    critic_out_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic output mlp."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    module_types: list[Literal["mlp", "conv"]] = None
    """Specifies the type of module for each sub module."""


@configclass
class RslRlPpoActorCriticBetaLidarTemporalCfg:
    """Configuration for the PPO actor-critic networks. Actor and critic networks have the same architecture."""

    class_name: str = "ActorCriticBetaLidarTemporal"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    non_lidar_dim: int = MISSING
    """The non-lidar dimensions."""

    lidar_dim: int = MISSING
    """The lidar dimensions."""

    lidar_extra_dim: int = MISSING
    """The extra lidar dimensions, ie relative position."""

    history_length: int = MISSING
    """The history length."""

    non_lidar_layer_dims: list[int] = MISSING
    """The hidden dimensions of the pre non-lidar layers."""

    lidar_compress_layer_dims: list[int] = MISSING
    """The hidden dimensions of the lidar compress layers, shared across history."""

    lidar_extra_in_dims: list[int] = MISSING
    """The hidden dimensions of the lidar extra input layers, shared across history."""

    lidar_extra_merge_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the lidar + extras merger mlp."""

    history_processing_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the history processing mlp."""

    gru_dim: int | None = None
    """The hidden dimensions of the GRU layer, if desired."""

    out_layer_dims: list[int] = MISSING
    """The hidden dimensions of the output layers."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticBetaRecurrentLidarCfg:
    """Configuration for the PPO actor-critic networks. Actor and critic networks have the same architecture."""

    class_name: str = "ActorCriticBetaRecurrentLidar"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    non_lidar_dim: int = MISSING
    """The non-lidar dimensions."""

    lidar_dim: int = MISSING
    """The lidar dimensions."""

    non_lidar_layer_dims: list[int] = MISSING
    """The hidden dimensions of the pre non-lidar layers."""

    lidar_compress_layer_dims: list[int] = MISSING
    """The hidden dimensions of the lidar compress layers, shared across history."""

    history_processing_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the history processing mlp."""

    gru_dim: int | None = None
    """The hidden dimensions of the GRU layer, if desired."""

    gru_layers: int = MISSING
    """The number of layers of the GRU."""

    out_layer_dims: list[int] = MISSING
    """The hidden dimensions of the output layers."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticBetaRecurrentLidarCnnCfg:
    """Configuration for the PPO actor-critic networks. Actor and critic networks have the same architecture."""

    class_name: str = "ActorCriticBetaRecurrentLidarCnn"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    non_lidar_dim: int = MISSING
    """The non-lidar dimensions."""

    lidar_dim: int = MISSING
    """The lidar dimensions."""

    lidar_extra_dim: int = MISSING
    """The extra lidar dimensions, ie relative position."""

    num_lidar_channels: int = MISSING
    """The number of channels in the lidar input."""

    non_lidar_layer_dims: list[int] = MISSING
    """The hidden dimensions of the pre non-lidar layers."""

    lidar_compress_conv_layer_dims: list[int] = MISSING
    """The hidden dimensions of the lidar compress cnn layers, shared across history."""

    lidar_compress_conv_kernel_sizes: list[int] = MISSING
    """The kernel sizes of the lidar compress cnn layers, shared across history."""

    lidar_compress_conv_strides: list[int] = MISSING
    """The strides of the lidar compress cnn layers, shared across history."""

    lidar_compress_conv_to_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the lidar compress cnn to mlp layers, shared across history."""

    lidar_extra_in_dims: list[int] = MISSING
    """The hidden dimensions of the lidar extra input layers, shared across history."""

    lidar_merge_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the lidar + extras merger mlp."""

    history_processing_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the history processing mlp."""

    gru_dim: int | None = None
    """The hidden dimensions of the GRU layer, if desired."""

    gru_layers: int = MISSING
    """The number of layers of the GRU."""

    out_layer_dims: list[int] = MISSING
    """The hidden dimensions of the output layers."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticBetaRecurrentLidarHeightCnnCfg:
    """Configuration for the PPO actor-critic networks. Actor and critic networks have the same architecture."""

    class_name: str = "ActorCriticBetaRecurrentLidarHeightCnn"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    non_lidar_dim: int = MISSING
    """The non-lidar dimensions."""

    lidar_dim: int = MISSING
    """The lidar dimensions."""

    heigh_scan_dims: tuple[int, int, int] = MISSING
    """The height scan dimensions, x, y, channels."""

    lidar_extra_dim: int = MISSING
    """The extra lidar dimensions, ie relative position."""

    num_lidar_channels: int = MISSING
    """The number of channels in the lidar input."""

    non_lidar_layer_dims: list[int] = MISSING
    """The hidden dimensions of the pre non-lidar layers."""

    lidar_compress_conv_layer_dims: list[int] = MISSING
    """The hidden dimensions of the lidar compress cnn layers, shared across history."""

    lidar_compress_conv_kernel_sizes: list[int] = MISSING
    """The kernel sizes of the lidar compress cnn layers, shared across history."""

    lidar_compress_conv_strides: list[int] = MISSING
    """The strides of the lidar compress cnn layers, shared across history."""

    lidar_compress_conv_to_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the lidar compress cnn to mlp layers, shared across history."""

    lidar_extra_in_dims: list[int] = MISSING
    """The hidden dimensions of the lidar extra input layers, shared across history."""

    lidar_merge_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the lidar + extras merger mlp."""

    height_conv_channels: list[int] = MISSING
    """The number of channels in the height conv input."""

    height_conv_kernel_sizes: list[int] = MISSING
    """The kernel sizes of the height conv layers."""

    height_conv_strides: list[int] = MISSING
    """The strides of the height conv layers."""

    heigh_conv_max_pool_kernel_sizes: list[int] = MISSING
    """The kernel sizes of the height conv max pool layers."""

    height_conv_to_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the height conv to mlp layers."""

    history_processing_mlp_dims: list[int] = MISSING
    """The hidden dimensions of the history processing mlp."""

    gru_dim: int | None = None
    """The hidden dimensions of the GRU layer, if desired."""

    gru_layers: int = MISSING
    """The number of layers of the GRU."""

    out_layer_dims: list[int] = MISSING
    """The hidden dimensions of the output layers."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticBetaCompressTemporalCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticBetaCompressTemporal"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    input_dims: list[int] = MISSING
    """The input dimensions of the sub mlps."""

    single_dims: list[int] = MISSING
    """Dimension of one timeframe input"""

    n_parallels: list[int] = MISSING
    """Number of parallel timeframes"""

    actor_siamese_hidden_dims_per_split: list[list[int]] = MISSING
    """The hidden dimensions of the actor sub mpls."""

    critic_siamese_hidden_dims_per_split: list[list[int]] = MISSING
    """The hidden dimensions of the critic sub mlps."""

    actor_out_hidden_dims_per_split: list[list[int]] = MISSING
    """The hidden out dimensions of the actor sub mpls."""

    critic_out_hidden_dims_per_split: list[list[int]] = MISSING
    """The hidden out dimensions of the critic sub mlps."""

    actor_out_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor output mlp."""

    critic_out_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic output mlp."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentBetaCfg:
    """Configuration for the PPO actor-critic recurrent networks."""

    class_name: str = "ActorCriticRecurrentBeta"
    """The policy class name. Default is ActorCritic."""

    rnn_type: str = MISSING
    """The type of RNN to use."""

    rnn_hidden_size: int = MISSING
    """The hidden size of the RNN."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg:
    """Configuration for the PPO actor-critic recurrent networks."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    rnn_type: str = MISSING
    """The type of RNN to use."""

    rnn_hidden_size: int = MISSING
    """The hidden size of the RNN."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda"
    """The device for the rl-agent. Default is cuda."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "orbit"
    """The neptune project name. Default is "orbit"."""

    wandb_project: str = "orbit"
    """The wandb project name. Default is "orbit"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlPpoSimpleNavTeacherCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "SimpleNavPolicy"
    """The policy class name. Default is ActorCritic."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    scan_cnn_channels: list[int] = MISSING
    scan_cnn_fc_shape: list[int] = MISSING
    scan_latent_size: int = MISSING

    # 1D conv for short history,
    history_channels: list[int] = MISSING
    history_kernel_size: int = MISSING
    history_fc_shape: list[int] = MISSING
    history_latent_size: int = MISSING
    output_mlp_size: list[int] = MISSING

    # input size information
    history_shape: list[int] = MISSING
    height_map_shape: list[int] = MISSING
    proprioceptive_shape: int = MISSING


@configclass
class RslRlPpoLocalNavBetaCfg:
    model_class: str = "BetaDistribution"
    scale: float = 3.0  # initial scale to alpha + beta
    num_logits: int = 6  # 2 x action dim


@configclass
class RslRlPpoLocalNavACCfg:
    class_name: str = "ActorCriticSeparate"
    actor_architecture: RslRlPpoSimpleNavTeacherCfg = MISSING
    critic_architecture: RslRlPpoSimpleNavTeacherCfg = MISSING
    action_distribution: RslRlPpoLocalNavBetaCfg = MISSING


#########################################################
# New implementation for the Crowd Navigation Task
#########################################################

@configclass
class RslRlPpoActorCriticBetaLidarCNNCfg:
    """Configuration for the PPO actor-critic networks. Actor and critic networks have the same architecture."""

    class_name: str = "ActorCriticBetaLidarCNN"
    """The policy class name. Default is ActorCritic."""

    beta_initial_logit: float = MISSING
    """The initial mean of the beta distribution."""

    beta_initial_scale: float = MISSING
    """The initial scale of the beta distribution."""

    target_dim: int = MISSING
    """The target position dimensions."""

    cpg_dim: int = MISSING
    """The cpg state dimensions."""

    lidar_dim: int = MISSING
    """The lidar dimensions."""

    lidar_extra_dim: int = MISSING
    """The extra lidar dimensions, ie relative position."""

    lidar_history_dim: int = MISSING
    """The number of history timesteps in the lidar input."""

    target_cpg_layer_dim: list[int] = MISSING
    """The hidden dimensions of the target position and cpg state."""

    lidar_cnn_layer_dim: list[int] = MISSING
    """The hidden dimensions of the lidar compress cnn layers, shared across history."""

    lidar_cnn_kernel_sizes: list[int] = MISSING
    """The kernel sizes of the lidar compress cnn layers, shared across history."""

    lidar_cnn_strides: list[int] = MISSING
    """The strides of the lidar compress cnn layers, shared across history."""

    lidar_cnn_to_mlp_layer_dim: list[int] = MISSING
    """The hidden dimensions of the lidar compress cnn to mlp layers, shared across history."""

    lidar_extra_mlp_layer_dim: list[int] = MISSING
    """The hidden dimensions of the lidar extra input layers, shared across history."""

    lidar_merge_mlp_layer_dim: list[int] = MISSING
    """The hidden dimensions of the lidar + extras merger mlp."""

    out_layer_dim: list[int] = MISSING
    """The hidden dimensions of the output layers."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""