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
    bootstrap_hamming_filter: float = 0.90
    bootstrap_property_filter: float = None
    bootstrap_kmeans_filter: bool = True
    bootstrap_max_count: int = 1

    train_batch_size: int = 32  # training batch size
    steps: int = 20000 * (train_batch_size // 32)  # training steps

    # save_image_interval: int = 200 * train_batch_size  # save images interval
    # save_model_interval: int = 1000 * train_batch_size  # save models interval
    # eval_playable_interval: int = 200 * train_batch_size  # check playable interval
    # bootstrap_interval: int = 40 * train_batch_size  # bootstrap
    dataset_size: int = 100

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005

    # others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda
    gpu_id: int = 5  # gpu index

    eval_playable_counts: int = 300  # number of z to check playable.
    clone_data: bool = False
    flip_data: bool = False
    bootstrap: str = "none"  # ["none", "random", "smart"]
    dataset_max_change_count: int = 5

    final_evaluation_levels: int = 10000

    def set_env(self):
        env = Env(self.env_name, self.env_version)
        self.env_fullname: str = f'{self.env_name}_{self.env_version}'
        self.input_shape = env.state_shape
        self.model_shapes = env.model_shape
        # data path
        self.level_data_path: str = (
            os.path.dirname(__file__) + "/data/level/" + self.env_fullname
        )  # Training dataset path
        self.checkpoints_path: str = (
            os.path.dirname(__file__) + "/checkpoints/" + self.env_fullname
        )  # save model path
        if self.env_name == 'mario':
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
    lambda_div: float = 50.0
    div_loss_threshold_playability: float = 0.0

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    train_batch_size: int = 32  # training batch size
    steps: int = 1000000  # training steps

    eval_playable_counts: int = 100  # number of z to check playable.
    save_image_interval: int = 200 * \
        train_batch_size  # save images interval
    # save_model_interval: int = 1000000  # save models interval
    # eval_playable_interval: int = 200 * train_batch_size  # check playable interval
    # bootstrap_interval: int = 20 * train_batch_size  # bootstrap
    dataset_size: int = 35
    save_image_epoch: int = 10
    save_model_epoch: int = 100
    eval_epoch: int = 10
    bootstrap_epoch: int = 1

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    bootstrap_filter = 0.90
    bootstrap_max_count: int = 10
    add_generated_max_count: int = 10
    reset_weight_bootstrap_count: int = 50
    reset_weight_interval: int = 2500 * train_batch_size
    reset_train_dataset_th: int = 200
    stop_generate_count = 2000


@dataclass
class SmallModelConfig(BaseConfig):
    dataset_type: str = "train"  # [train, generated]

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
    model_type: str = "sa"  # "normal","simple","branch","small"
    use_self_attention_g: list[int] = field(default_factory=lambda: [1, 2])
    use_self_attention_d: list[int] = field(default_factory=lambda: [0, 1])
    use_linear4z2features_g: bool = False
    use_deconv_g: bool = False
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
    bootstrap_hamming_filter: float = 0.90
    bootstrap_property_filter: float = None
    bootstrap_kmeans_filter: bool = True
    bootstrap_max_count: int = 10

    train_batch_size: int = 32  # training batch size
    steps: int = 10000 * (train_batch_size // 32)  # training steps
    dataset_size: int = 35

    save_image_epoch: int = 100
    save_model_epoch: int = 1000
    eval_epoch: int = 100
    bootstrap_epoch: int = 1

    # save_image_interval: int = 100 * train_batch_size  # save images interval
    # save_model_interval: int = 1000 * train_batch_size  # save models interval
    # eval_playable_interval: int = 100 * train_batch_size  # check playable interval
    # bootstrap_interval: int = 10 * train_batch_size  # bootstrap

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005
