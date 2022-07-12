from dataclasses import dataclass, field
import os


@dataclass
class TrainingConfig:
    name: str = "none"

    ### environment
    env_name: str = "zelda"

    ### data path
    level_data_path: str = (
        os.path.dirname(__file__) + "/data/level"
    )  # Training dataset path
    checkpoints_path: str = (
        os.path.dirname(__file__) + "/checkpoints"
    )  # save model path

    ### model define
    latent_size: int = 128  # latent dims for generation
    generator_filters: int = 256
    discriminator_filters: int = 16
    input_shape: tuple = (8, 12, 16)
    model_shapes: list = field(default_factory=lambda: [(3, 4), (6, 8), (12, 16)])
    is_self_attention_g: bool = True
    is_self_attention_d: bool = True
    is_minibatch_std: bool = False
    is_spectral_norm: bool = False
    is_conditional: bool = False

    ### learning parameters
    adv_loss: str = "baseline"  # ["baseline","hinge"]
    div_loss: str = "l2"  # ["l1","l2","none"]
    lambda_div = 5.0

    generator_lr: float = 0.0001
    discriminator_lr: float = 0.0001
    epochs: int = 1000000  # training epochs
    steps: int = 50000  # training steps
    train_batch_size: int = 64  # training batch size
    test_batch_size: int = 5  # test batch size
    save_image_interval_epoch: int = 10  # save images interval
    save_model_interval_epoch: int = 500  # save models interval
    eval_playable_interval_epoch: int = 10  # check playable interval

    ### others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda

    eval_playable_counts = 300  # number of z to check playable.
    extend_data = True
    flip_data = True
    bootstrap: bool = True


@dataclass
class EvaluationConfig:
    name: str = "none"

    ### environment
    env_name: str = "zelda"

    ### data path
    level_data_path: str = (
        os.path.dirname(__file__) + "/data/level"
    )  # Training dataset path
    checkpoints_path: str = (
        os.path.dirname(__file__) + "/checkpoints"
    )  # save model path

    ### model define
    latent_size: int = 128  # latent dims for generation
    generator_filters: int = 256
    discriminator_filters: int = 16
    input_shape: tuple = (8, 12, 16)
    model_shapes: list = field(default_factory=lambda: [(3, 4), (6, 8), (12, 16)])
    is_self_attention_g: bool = True
    is_minibatch_std: bool = False
    is_spectral_norm: bool = False
    is_conditional: bool = False

    ### others parameters
    seed: int = 0  # random seed
    cuda: bool = True  # use cuda

    eval_playable_counts = 1000  # number of z to check playable.
