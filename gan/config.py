from __future__ import annotations
from dataclasses import dataclass, field
from .game.env import Game
import os


@dataclass
class BaseConfig:
    # training name
    name: str = "journal"

    # environment name
    env_name: str = "zelda"
    env_version: str = 'v1'

    # model architecture
    latent_size: int = 32  # latent dims for generation
    generator_filters: int = 64
    discriminator_filters: int = 64
    use_self_attention_g: bool = False
    use_self_attention_d: bool = False
    use_spectral_norm_d: bool = True
    normalization_d: str = None  # implemented None, "Batch", "Instance" or "Layer"
    use_clipping_d: bool = True
    extra_layers_g: int = 0
    extra_layers_d: int = 0
    # use_linear4z2features_g: bool = False
    # use_deconv_g: bool = True
    # use_bn_d: bool = False
    # use_sn_d: bool = False
    # use_pooling_d: bool = False
    # use_minibatch_std: bool = False
    # use_spectral_norm: bool = False
    # use_conditional: bool = False

    # learning parameters
    adv_loss: str = "hinge"  # ["baseline","hinge",'wgan']
    div_loss: str = None  # ["l1","l2",'l1-latent',None]
    lambda_div: float = 0.0
    use_recon_loss: bool = False
    recon_lambda: float = 1.0
    use_gradient_penalty: bool = False
    gp_lambda: float = 0.5
    generator_lr: float = 0.0001
    discriminator_lr: float = 0.0001
    discriminator_update_count: int = 5
    train_batch_size: int = 32  # training batch size
    steps: int = 5000  # training steps

    # dataset parameters
    bootstrapping_mode: str = None  # [None, "regacy", "proposal"]
    dataset_size: int = 5
    clone_data: bool = False
    flip_data: bool = False
    initial_data_featuring: bool = False
    bootstrap_hamming_filter: float = None
    select_newlevels_mode: str = 'kmeans'  # ['kmeans','features']
    diversity_sampling_mode: str = 'legacy'
    eval_playable_counts: int = 128  # number of z to check playable.
    final_evaluation_levels: int = 15000

    # others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda
    gpu_id: int = 3  # gpu index
    save_model_interval: int = 10000
    save_image_interval: int = 100
    eval_interval: int = 100
    bootstrapping_interval: int = 10
    bootstrapping_steps: int = steps

    def set_env(self, game: Game):
        self.env_name = game.name
        self.env_version = game.version
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

        # generated level data path (only use in DataExtendConfig)
        self.generated_data_path: str = (
            os.path.dirname(__file__) + "/data/level/" +
            self.env_fullname + '/generated/'
        )

        # checkpoint path
        self.checkpoints_path: str = (
            os.path.dirname(__file__) + "/checkpoints/" + self.env_fullname
        )  # save model path

    def set_augmentation(self):
        self.is_augmentation = True
        self.bootstrapping_interval = 5
        self.steps = 1000000
        self.reset_weight_epoch: int = 1000
        self.reset_weight_bootstrap_count: int = 50
        self.reset_train_dataset_th: int = 200
        self.stop_generate_count = 1500

    def set_learning_from_augmented(self):
        self.level_data_path: str = (
            os.path.dirname(__file__) + "/data/level/" +
            self.env_fullname + '/generated/'
        )  # Training dataset path


@dataclass
class MarioConfig(BaseConfig):
    dataset_size: int = 10


@dataclass
class ZeldaConfig(BaseConfig):
    pass


@dataclass
class BoulderdashConfig(BaseConfig):
    pass
