from __future__ import annotations
from dataclasses import dataclass, field
from .env import Env
import os


@dataclass
class BaseConfig:
    # training name
    name: str = "none"
    # environment name
    env_name: str = "zelda"
    env_version: str = 'v1'
    env_fullname: str = f'{env_name}_{env_version}'
    # data path
    level_data_path: str = (
        os.path.dirname(__file__) + "/data/level"
    )  # Training dataset path
    checkpoints_path: str = (
        os.path.dirname(__file__) + "/checkpoints"
    )  # save model path
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
    steps: int = 20000 * (train_batch_size // 32)  # training steps

    save_image_interval: int = 200 * train_batch_size  # save images interval
    save_model_interval: int = 1000 * train_batch_size  # save models interval
    eval_playable_interval: int = 200 * train_batch_size  # check playable interval
    bootstrap_interval: int = 40 * train_batch_size  # bootstrap
    dataset_size: int = 50

    use_recon_loss: bool = True
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005

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
            self.env_version = 'v0'
            self.checkpoints_path = (
                os.path.dirname(__file__) + "/checkpoints_mario"
            )  # save model path
            self.generator_filters = 64
            self.discriminator_filters = 64
        else:
            pass


@dataclass
class DataExtendConfig(BaseConfig):
    dataset_type: str = "train"  # [train, generated]
    seed: int = 0

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
    save_image_interval: int = 200 * \
        train_batch_size  # save images interval
    save_model_interval: int = 1000000  # save models interval
    eval_playable_interval: int = 200 * train_batch_size  # check playable interval
    bootstrap_interval: int = 20 * train_batch_size  # bootstrap
    dataset_size: int = 100

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    bootstrap_filter = 0.90
    bootstrap_max_count: int = 1
    add_generated_max_count: int = 1
    reset_weight_bootstrap_count: int = 3
    reset_weight_interval: int = 3000 * train_batch_size
    reset_train_dataset_th: int = 30
    stop_generate_count = 200


@dataclass
class SmallModelConfig(BaseConfig):
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
    steps: int = 20000 * (train_batch_size // 32)  # training steps

    save_image_interval: int = 200 * train_batch_size  # save images interval
    save_model_interval: int = 1000 * train_batch_size  # save models interval
    eval_playable_interval: int = 200 * train_batch_size  # check playable interval
    bootstrap_interval: int = 40 * train_batch_size  # bootstrap
    dataset_size: int = 50

    use_recon_loss: bool = True
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005
