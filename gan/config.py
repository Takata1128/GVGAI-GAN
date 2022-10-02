from __future__ import annotations
from dataclasses import dataclass, field
from .env import Env
import os


@dataclass
class TrainingConfig:
    # training name
    name: str = "none"

    # environment name
    env_name: str = "zelda"
    env_version: str = 'v1'

    # data path
    level_data_path: str = (
        os.path.dirname(__file__) + "/data/level"
    )  # Training dataset path

    dataset_type: str = "generated_part"  # [train, generated]

    checkpoints_path: str = (
        os.path.dirname(__file__) + "/checkpoints"
    )  # save model path

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
    input_shape: tuple[int] = None
    model_shapes: list[tuple[int]] = None
    model_type: str = "small"  # "normal","simple","branch","small"
    use_self_attention_g: list[int] = field(default_factory=lambda: [1, 2])
    use_self_attention_d: list[int] = field(default_factory=lambda: [0, 1])
    use_deconv_g: bool = True
    use_bn_d: bool = False
    use_sn_d: bool = True
    use_minibatch_std: bool = False
    use_spectral_norm: bool = False
    use_conditional: bool = False

    # learning parameters
    adv_loss: str = "hinge"  # ["baseline","hinge"]
    div_loss: str = "none"  # ["l1","l2","none"]
    lambda_div: float = 50.0
    div_loss_threshold_playability: float = 0.0

    train_batch_size: int = 64  # training batch size
    steps: int = 50000  # training steps

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005
    epochs: int = 1000000  # training epochs
    smooth_label_value: float = 0.0  # prob of flipping real label
    save_image_interval: int = 50000  # save images interval
    save_model_interval: int = 1000000  # save models interval
    eval_playable_interval: int = 50000  # check playable interval

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    # others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda
    gpu_id: int = 5  # gpu index

    eval_playable_counts: int = 300  # number of z to check playable.
    clone_data: bool = False
    dataset_size: int = 512
    flip_data: bool = False
    bootstrap: str = "none"  # ["none", "random", "smart"]
    dataset_max_change_count: int = 5

    def set_env(self):
        env = Env(self.env_name, self.env_version)
        self.input_shape = env.state_shape
        self.model_shapes = env.model_shape
        if self.env_name == 'mario':
            self.generator_filters = 64
            self.discriminator_filters = 64
        else:
            pass


@dataclass
class DataExtendConfig(TrainingConfig):
    dataset_type: str = "train"  # [train, generated]
    seed: int = 0

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
    input_shape: tuple[int] = None
    model_shapes: list[tuple[int]] = None
    model_type: str = "small"  # "normal","simple","branch","small"
    use_self_attention_g: list[int] = field(default_factory=lambda: [1, 2])
    use_self_attention_d: list[int] = field(default_factory=lambda: [0, 1])
    use_linear4z2features_g: bool = False
    use_deconv_g: bool = True
    use_sn_d: bool = False
    use_bn_d: bool = False
    use_pooling_d: bool = False
    use_minibatch_std: bool = True
    use_spectral_norm: bool = False
    use_conditional: bool = False

    # learning parameters
    adv_loss: str = "hinge"  # ["baseline","hinge"]
    div_loss: str = "l1"  # ["l1","l2","none"]
    lambda_div: float = 10.0
    div_loss_threshold_playability: float = 0.0

    use_recon_loss: bool = True
    recon_lambda: float = 1.0

    train_batch_size: int = 32  # training batch size
    steps: int = 100000  # training steps

    eval_playable_counts: int = 100  # number of z to check playable.

    reset_weight_interval: int = 1000000 * train_batch_size
    reset_weight_threshold: float = 0.05

    save_image_interval: int = 200 * \
        train_batch_size  # save images interval
    save_model_interval: int = 1000000  # save models interval
    eval_playable_interval: int = 200 * train_batch_size  # check playable interval
    bootstrap_interval: int = 20*train_batch_size  # bootstrap
    dataset_size: int = 100

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    bootstrap_filter = 0.90
    bootstrap_max_count: int = 1
    add_generated_max_count: int = 1
    reset_weight_bootstrap_count: int = 3
    stop_generate_count = 150


@dataclass
class NormalModelConfig(TrainingConfig):
    dataset_type: str = "generated_base"  # [train, generated]

    # model define
    latent_size: int = 128  # latent dims for generation
    generator_filters: int = 256
    discriminator_filters: int = 16
    input_shape: tuple[int] = None
    model_shapes: list[tuple[int]] = None
    model_type: str = "normal"  # "normal","simple","branch","small"
    use_self_attention_g: bool = True
    use_self_attention_d: bool = True
    use_minibatch_std: bool = False
    use_spectral_norm: bool = False
    use_conditional: bool = False

    # learning parameters
    adv_loss: str = "baseline"  # ["baseline","hinge"]
    div_loss: str = "none"  # ["l1","l2","none"]
    lambda_div: float = 50.0
    div_loss_threshold_playability: float = 0.0
    dataset_size: int = 64

    bootstrap: str = "none"  # ["none", "random", "smart"]

    train_batch_size: int = 64  # training batch size
    steps: int = 50000  # training steps


@dataclass
class SmallModelConfig(TrainingConfig):
    dataset_type: str = "train"  # [train, generated]

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
    model_type: str = "small"  # "normal","simple","branch","small"
    use_self_attention_g: list[int] = field(default_factory=lambda: [1, 2])
    use_self_attention_d: list[int] = field(default_factory=lambda: [0, 1])
    use_linear4z2features_g: bool = False
    use_deconv_g: bool = True
    use_bn_d: bool = False
    use_sn_d: bool = False
    use_pooling_d: bool = False
    use_minibatch_std: bool = False
    use_spectral_norm: bool = False
    use_conditional: bool = False

    # learning parameters
    adv_loss: str = "hinge"  # ["baseline","hinge"]
    div_loss: str = "l1"  # ["l1","l2","none"]
    lambda_div: float = 50.0
    div_loss_threshold_playability: float = 0.0

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    bootstrap_filter: float = 0.90
    bootstrap_max_count: int = 1

    train_batch_size: int = 32  # training batch size
    steps: int = 20000*(train_batch_size//32)  # training steps

    save_image_interval: int = 200*train_batch_size  # save images interval
    save_model_interval: int = 1000*train_batch_size  # save models interval
    eval_playable_interval: int = 200*train_batch_size  # check playable interval
    bootstrap_interval: int = 40*train_batch_size  # bootstrap
    dataset_size: int = 50

    use_recon_loss: bool = True
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005


@dataclass
class SAModelConfig(TrainingConfig):
    dataset_type: str = "train"  # [train, generated]

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 64
    discriminator_filters: int = 256
    model_type: str = "sa"  # "normal","small","sa"
    use_self_attention_g: bool = True
    use_self_attention_d: bool = True
    use_deconv_g: bool = True
    use_bn_d: bool = False
    use_sn_d: bool = False
    use_minibatch_std: bool = True
    use_spectral_norm: bool = False
    use_conditional: bool = False

    # learning parameters
    adv_loss: str = "hinge"  # ["baseline","hinge"]
    div_loss: str = "none"  # ["l1","l2","none"]
    lambda_div: float = 10.0
    div_loss_threshold_playability: float = 0.0

    bootstrap: str = "none"  # ["none", "random", "smart"]

    train_batch_size: int = 32  # training batch size
    steps: int = 50000*(train_batch_size//32)  # training steps

    save_image_interval: int = 1000*32  # save images interval
    save_model_interval: int = 5000*32  # save models interval
    eval_playable_interval: int = 1000*32  # check playable interval

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005

    def set_env(self):
        env = Env(self.env_name, self.env_version)
        self.input_shape = env.state_shape
        self.model_shapes = env.model_shape


@dataclass
class EvaluationConfig:
    name: str = "none"

    # environment
    env_name: str = "zelda"

    # data path
    level_data_path: str = (
        os.path.dirname(__file__) + "/data/level"
    )  # Training dataset path
    checkpoints_path: str = (
        os.path.dirname(__file__) + "/checkpoints"
    )  # save model path

    # model define
    latent_size: int = 128  # latent dims for generation
    generator_filters: int = 256
    discriminator_filters: int = 16
    input_shape: tuple = (8, 12, 16)
    model_shapes: list = field(default_factory=lambda: [
                               (3, 4), (6, 8), (12, 16)])
    is_self_attention_g: bool = True
    is_minibatch_std: bool = False
    is_spectral_norm: bool = False
    is_conditional: bool = False

    # others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda

    eval_playable_counts = 1000  # number of z to check playable.
