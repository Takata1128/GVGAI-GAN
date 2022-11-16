from __future__ import annotations
from dataclasses import dataclass, field
from .game.env import Game
import os


@dataclass
class BaseConfig:
    # training name
    name: str = "none"
    # environment name
    env_name: str = "zelda"
    env_version: str = 'v1'

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
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
    use_diversity_sampling: bool = False
    bootstrap_max_count: int = 1

    train_batch_size: int = 32  # training batch size
    steps: int = 10000 * (train_batch_size // 32)  # training steps

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
    gpu_id: int = 4  # gpu index

    eval_playable_counts: int = 300  # number of z to check playable.
    clone_data: bool = False
    flip_data: bool = False
    bootstrap: str = None
    dataset_max_change_count: int = 5
    final_evaluation_levels: int = 10000

    def set_env(self):
        self.env_fullname: str = f'{self.env_name}_{self.env_version}'

        # data path
        self.data_path: str = (
            os.path.dirname(__file__) + '/data/level/' +
            self.env_fullname + '/'
        )

        # training level data path
        self.level_data_path: str = (
            os.path.dirname(__file__) + "/data/level/" +
            self.env_fullname + '/train/'
        )  # Training dataset path
        # generated data path (only use in DataExtendConfig)

        # generated level data path
        self.generated_data_path: str = (
            os.path.dirname(__file__) + "/data/level/" +
            self.env_fullname + '/generated/'
        )

        # checkpoint path
        self.checkpoints_path: str = (
            os.path.dirname(__file__) + "/checkpoints/" + self.env_fullname
        )  # save model path


@dataclass
class DataExtendConfig(BaseConfig):
    dataset_type: str = "train"  # [train, generated]
    seed: int = 0

    # learning parameters
    adv_loss: str = "hinge"  # ["baseline","hinge"]
    div_loss: str = 'l1'  # ["l1","l2","none"]
    lambda_div: float = 50.0
    div_loss_threshold_playability: float = 0.0

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    train_batch_size: int = 32  # training batch size
    steps: int = 1000000  # training steps

    eval_playable_counts: int = 100  # number of z to check playable.
    save_image_epoch: int = 10
    save_model_epoch: int = 100
    reset_weight_epoch: int = 5000
    eval_epoch: int = 10
    bootstrap_epoch: int = 1

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    bootstrap_filter = 0.90
    bootstrap_max_count: int = 10
    use_diversity_sampling: bool = True

    add_generated_max_count: int = 10
    reset_weight_bootstrap_count: int = 50
    reset_train_dataset_th: int = 200
    stop_generate_count = 1500


@dataclass
class SmallModelConfig(BaseConfig):
    dataset_type: str = "train"  # [train, generated]

    # model define
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 128
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

    bootstrap: str = "smart"  # ["none", "random", "smart"]
    bootstrap_hamming_filter: float = 0.90
    bootstrap_property_filter: float = None
    bootstrap_kmeans_filter: bool = True
    bootstrap_max_count: int = 10
    use_diversity_sampling: bool = True

    train_batch_size: int = 32  # training batch size
    steps: int = 10000 * (train_batch_size // 32)  # training steps

    save_image_epoch: int = 100
    save_model_epoch: int = 1000
    eval_epoch: int = 100
    bootstrap_epoch: int = 1

    use_recon_loss: bool = False
    recon_lambda: float = 1.0

    generator_lr: float = 0.00005
    discriminator_lr: float = 0.00005
