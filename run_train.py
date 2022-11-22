import argparse
from gan.game.zelda import Zelda
from gan.game.mario import Mario
from gan.trainer import Trainer
from gan.level_dataset_extend import prepare_dataset
from gan.models.dcgan_models import DCGAN_G, DCGAN_D
from gan.models.small_models import Generator, Discriminator
import gan.config as cfg
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    def get_models(game, config, device):
        latent_shape = (config.latent_size,)
        generator = Generator(
            out_ch=game.input_shape[0],
            shapes=game.model_shape,
            z_shape=latent_shape,
            filters=config.generator_filters,
            use_linear4z2features_g=config.use_linear4z2features_g,
            use_self_attention=config.use_self_attention_g,
            use_conditional=config.use_conditional,
            use_deconv_g=config.use_deconv_g
        ).to(device)
        discriminator = Discriminator(
            in_ch=game.input_shape[0],
            shapes=game.model_shape[::-1],
            filters=config.discriminator_filters,
            use_bn=config.use_bn_d,
            use_self_attention=config.use_self_attention_d,
            use_minibatch_std=config.use_minibatch_std,
            use_recon_loss=config.use_recon_loss,
            use_conditional=config.use_conditional,
            use_sn=config.use_sn_d,
            use_pooling=config.use_pooling_d
        ).to(device)
        models_dict = {
            'generator': generator,
            'discriminator': discriminator
        }
        return models_dict

    # Hinge + Conventional Bootstrap
    game = Zelda('zelda', 'v1')
    config = cfg.ZeldaConfig()
    config.set_env()
    config.bootstrap = 'baseline'
    config.adv_loss = 'hinge'
    config.div_loss = None
    config.discriminator_lr = 0.0004
    config.generator_lr = 0.0001
    config.use_diversity_sampling = False
    if config.dataset_type == "train":
        prepare_dataset(
            game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models_dict = get_models(game, config, device)
    trainer = Trainer(game, models_dict, config)
    trainer.train()

    # Hinge + Proposed
    game = Zelda('zelda', 'v1')
    config = cfg.ZeldaConfig()
    config.set_env()
    config.bootstrap = 'smart'
    config.bootstrap_hamming_filter = 0.90
    config.adv_loss = 'hinge'
    config.div_loss = 'l1'
    config.use_diversity_sampling = False
    config.lambda_div = 50.0
    if config.dataset_type == "train":
        prepare_dataset(
            game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models_dict = get_models(game, config, device)
    trainer = Trainer(game, models_dict, config)
    trainer.train()

    # Hinge + Proposed
    game = Zelda('zelda', 'v1')
    config = cfg.ZeldaConfig()
    config.set_env()
    config.bootstrap = 'smart'
    config.bootstrap_hamming_filter = 0.90
    config.adv_loss = 'hinge'
    config.div_loss = 'l1'
    config.use_diversity_sampling = True
    config.lambda_div = 50.0
    if config.dataset_type == "train":
        prepare_dataset(
            game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models_dict = get_models(game, config, device)
    trainer = Trainer(game, models_dict, config)
    trainer.train()

    # Hinge + Conventional Bootstrap
    game = Zelda('zelda', 'v1')
    config = cfg.ZeldaConfig()
    config.set_env()
    config.bootstrap = 'baseline'
    config.adv_loss = 'baseline'
    # config.use_gradient_penalty = True
    # config.discrimianator_update_count = 5
    config.use_diversity_sampling = True
    config.div_loss = None
    if config.dataset_type == "train":
        prepare_dataset(
            game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models_dict = get_models(game, config, device)
    trainer = Trainer(game, models_dict, config)
    trainer.train()

    # Hinge + Proposed
    game = Zelda('zelda', 'v1')
    config = cfg.ZeldaConfig()
    config.set_env()
    config.bootstrap = 'smart'
    config.bootstrap_hamming_filter = 0.90
    config.adv_loss = 'baseline'
    # config.discrimianator_update_count = 5
    # config.use_gradient_penalty = True
    config.use_diversity_sampling = True
    config.div_loss = 'l1'
    config.lambda_div = 50.0
    if config.dataset_type == "train":
        prepare_dataset(
            game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models_dict = get_models(game, config, device)
    trainer = Trainer(game, models_dict, config)
    trainer.train()

    # # Wgan + Conventional Bootstrap
    # game = Zelda('zelda', 'v1')
    # config = cfg.ZeldaConfig()
    # config.set_env()
    # config.bootstrap = 'baseline'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # # Wgan + Proposed
    # game = Zelda('zelda', 'v1')
    # config = cfg.ZeldaConfig()
    # config.set_env()
    # config.bootstrap = 'smart'
    # config.bootstrap_hamming_filter = 0.90
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.div_loss = 'l1'
    # config.lambda_div = 0.5
    # config.use_diversity_sampling = True
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # losses = ['baseline', 'wgan', 'wgan-gp', 'lsgan', 'hinge']
    # d_updates = [1, 5]

    # for loss in losses:
    #     for d_update in d_updates:
    #         config = cfg.MarioConfig()
    #         config.env_name = "mario"
    #         config.env_version = 'v0'
    #         game = Mario(config.env_name, config.env_version)
    #         config.set_env()
    #         config.adv_loss = loss
    #         config.discrimianator_update_count = d_update
    #         device = torch.device(
    #             f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #         generator = DCGAN_G(32, config.latent_size,
    #                             game.input_shape[0], config.generator_filters)
    #         discriminator = DCGAN_D(32, config.latent_size,
    #                                 game.input_shape[0], config.discriminator_filters)
    #         models_dict = {
    #             'generator': generator,
    #             'discriminator': discriminator
    #         }
    #         prepare_dataset(
    #             game=game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size
    #         )
    #         trainer = Trainer(game, models_dict, config)
    #         trainer.train()
