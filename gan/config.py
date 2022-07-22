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
    checkpoints_path: str = (
        os.path.dirname(__file__) + "/checkpoints"
    )  # save model path

    # model define
    latent_size: int = 64  # latent dims for generation
    generator_filters: int = 128
    discriminator_filters: int = 8
    input_shape: tuple[int] = None
    model_shapes: list[tuple[int]] = None
    is_self_attention_g: bool = True
    is_self_attention_d: bool = True
    is_minibatch_std: bool = False
    is_spectral_norm: bool = False
    is_conditional: bool = False

    # learning parameters
    adv_loss: str = "baseline"  # ["baseline","hinge"]
    div_loss: str = "l1"  # ["l1","l2","none"]
    lambda_div: float = 100.0
    div_loss_threshold_playability: float = 0.5

    generator_lr: float = 0.0001
    discriminator_lr: float = 0.0001
    epochs: int = 1000000  # training epochs
    steps: int = 50000*4  # training steps
    train_batch_size: int = 16  # training batch size
    test_batch_size: int = 5  # test batch size
    label_flip_prob: float = 0.0  # prob of flipping real label
    save_image_interval_epoch: int = 100  # save images interval
    save_model_interval_epoch: int = 5000  # save models interval
    eval_playable_interval_epoch: int = 100  # check playable interval

    # others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda

    eval_playable_counts: int = 300  # number of z to check playable.
    clone_data: bool = False
    dataset_size: int = 512
    flip_data: bool = False
    bootstrap: str = "none"  # ["none", "random", "smart"]
    dataset_max_change_count: int = 300

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
