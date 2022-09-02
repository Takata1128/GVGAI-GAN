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
    latent_size: int = 128  # latent dims for generation
    generator_filters: int = 256
    discriminator_filters: int = 16
    input_shape: tuple[int] = None
    model_shapes: list[tuple[int]] = None
    model_type: str = "small"  # "normal","simple","branch","small"
    use_self_attention_g: bool = False
    use_self_attention_d: bool = False
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
    gpu_id: int = 0  # gpu index

    eval_playable_counts: int = 300  # number of z to check playable.
    clone_data: bool = False
    dataset_size: int = 512
    flip_data: bool = False
    bootstrap: str = "none"  # ["none", "random", "smart"]
    dataset_max_change_count: int = 5

    recall_weight_threshold: float = 10.05

    def set_env(self):
        env = Env(self.env_name, self.env_version)
        self.input_shape = env.state_shape
        self.model_shapes = env.model_shape


@dataclass
class DataExtendConfig(TrainingConfig):
    dataset_type: str = "train"  # [train, generated]

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 256
    discriminator_filters: int = 16
    input_shape: tuple[int] = None
    model_shapes: list[tuple[int]] = None
    model_type: str = "normal"  # "normal","simple","branch","small"
    use_self_attention_g: bool = True
    use_self_attention_d: bool = True
    use_deconv_g: bool = False
    use_bn_d: bool = False
    use_minibatch_std: bool = False
    use_spectral_norm: bool = False
    use_conditional: bool = False

    save_image_interval_epoch: int = 100  # save images interval
    save_model_interval_epoch: int = 5000  # save models interval
    eval_playable_interval_epoch: int = 100  # check playable interval

    # learning parameters
    adv_loss: str = "baseline"  # ["baseline","hinge"]
    div_loss: str = "none"  # ["l1","l2","none"]
    lambda_div: float = 50.0
    div_loss_threshold_playability: float = 0.0

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    train_batch_size: int = 64  # training batch size
    steps: int = 100000  # training steps

    recall_weight_threshold: float = 0.05

    save_image_interval: int = 20000  # save images interval
    save_model_interval: int = 1000000  # save models interval
    eval_playable_interval: int = 20000  # check playable interval

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    dataset_max_change_count: int = 3


@dataclass
class NormalModelConfig(TrainingConfig):
    dataset_type: str = "generated_good"  # [train, generated]

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

    bootstrap: str = "none"  # ["none", "random", "smart"]

    train_batch_size: int = 64  # training batch size
    steps: int = 50000  # training steps


@dataclass
class SmallModelConfig(TrainingConfig):
    dataset_type: str = "generated_fixed"  # [train, generated]

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
    model_type: str = "small"  # "normal","simple","branch","small"
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

    use_recon_loss: bool = True
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005

    def set_env(self):
        env = Env(self.env_name, self.env_version)
        self.input_shape = env.state_shape
        self.model_shapes = env.model_shape


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
