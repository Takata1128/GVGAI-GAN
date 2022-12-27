import argparse
from gan.game.env import Game
from gan.game.zelda import Zelda
from gan.game.mario import Mario
from gan.game.boulderdash import Boulderdash
from gan.trainer import Trainer
from gan.models.general_models import Generator, Discriminator
import gan.config as cfg
import torch
import random
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    def get_models(game: Game, config: cfg.BaseConfig):
        # reproducible settings
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(
            f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
        generator = Generator(
            isize=game.input_shape[1], nz=config.latent_size, nc=game.input_shape[
                0], ngf=config.generator_filters, self_attention=config.use_self_attention_g, n_extra_layers=config.extra_layers_g
        ).to(device)
        discriminator = Discriminator(
            isize=game.input_shape[1], nz=config.latent_size, nc=game.input_shape[
                0], ndf=config.discriminator_filters, use_self_attention=config.use_self_attention_d, use_spectral_norm=config.use_spectral_norm_d, normalization=config.normalization_d, use_recon_loss=config.use_recon_loss, n_extra_layers=config.extra_layers_d
        ).to(device)
        models_dict = {
            'generator': generator,
            'discriminator': discriminator
        }
        return models_dict

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.seed = i
        config.initial_data_prob = 0.20
        config.initial_data_sampling_steps = 3000
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Zelda()
        config = cfg.ZeldaConfig()
        config.set_env(game)
        config.seed = i
        config.initial_data_prob = 0.20
        config.initial_data_sampling_steps = 3000
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Boulderdash()
        config = cfg.BoulderdashConfig()
        config.set_env(game)
        config.seed = i
        config.initial_data_prob = 0.20
        config.initial_data_sampling_steps = 3000
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

        # game = Mario()
        # config = cfg.MarioConfig()
        # config.set_env(game)
        # config.seed = i
        # config.bootstrap = None
        # config.div_loss = None
        # config.use_diversity_sampling = True
        # config.generator_lr = 0.0001
        # config.discriminator_lr = 0.0001
        # config.discriminator_update_count = 5
        # config.use_spectral_norm_d = True
        # config.use_clipping_d = True
        # config.use_gradient_penalty = False
        # config.normalization_d = None
        # config.set_learning_from_augmented()
        # models_dict = get_models(game, config)
        # trainer = Trainer(game, models_dict, config)
        # trainer.train()

    # game = Boulderdash()
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = None
    # config.div_loss = None
    # config.lambda_div = 10.0
    # config.use_diversity_sampling = False
    # config.generator_lr = 0.0001
    # config.discriminator_lr = 0.0001
    # config.discriminator_update_count = 5
    # config.use_spectral_norm_d = True
    # config.use_clipping_d = True
    # config.use_gradient_penalty = False
    # config.normalization_d = None
    # models_dict = get_models(game, config)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # for i in range(3):
    #     game = Boulderdash()
    #     config = cfg.BoulderdashConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = False
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i in range(3):
    #     game = Boulderdash()
    #     config = cfg.BoulderdashConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i in range(3):
    #     game = Boulderdash()
    #     config = cfg.BoulderdashConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'smart'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i in range(1):
    #     game = Boulderdash()
    #     config = cfg.BoulderdashConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1'
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # game = Zelda('v1')
    # config = cfg.ZeldaConfig()
    # config.set_env(game)
    # config.bootstrap = None
    # config.div_loss = None
    # config.lambda_div = 10.0
    # config.use_diversity_sampling = False
    # config.generator_lr = 0.0001
    # config.discriminator_lr = 0.0001
    # config.discriminator_update_count = 5
    # config.use_spectral_norm_d = True
    # config.use_clipping_d = True
    # config.use_gradient_penalty = False
    # config.normalization_d = None
    # models_dict = get_models(game, config)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # for i in range(3):
    #     game = Zelda('v1')
    #     config = cfg.ZeldaConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = False
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i in range(3):
    #     game = Zelda('v1')
    #     config = cfg.ZeldaConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # for i in range(3):
    # #     game = Zelda('v1')
    # #     config = cfg.ZeldaConfig()
    # #     config.set_env(game)
    # #     config.seed = i
    # #     config.bootstrap = 'smart'
    # #     config.div_loss = None
    # #     config.lambda_div = 10.0
    # #     config.use_diversity_sampling = True
    # #     config.generator_lr = 0.0001
    # #     config.discriminator_lr = 0.0001
    # #     config.discriminator_update_count = 5
    # #     config.use_spectral_norm_d = True
    # #     config.use_clipping_d = True
    # #     config.use_gradient_penalty = False
    # #     config.normalization_d = None
    # #     models_dict = get_models(game, config)
    # #     trainer = Trainer(game, models_dict, config)
    # #     trainer.train()

    # for i in range(3):
    #     game = Zelda('v1')
    #     config = cfg.ZeldaConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1'
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     config.bootstrap_hamming_filter = 0.90
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # game = Mario()
    # config = cfg.MarioConfig()
    # config.set_env(game)
    # config.bootstrap = None
    # config.div_loss = None
    # config.lambda_div = 10.0
    # config.use_diversity_sampling = False
    # config.generator_lr = 0.0001
    # config.discriminator_lr = 0.0001
    # config.discriminator_update_count = 5
    # config.use_spectral_norm_d = True
    # config.use_clipping_d = True
    # config.use_gradient_penalty = False
    # config.normalization_d = None
    # config.dataset_size = 10
    # models_dict = get_models(game, config)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # for i in range(3):
    #     game = Mario()
    #     config = cfg.MarioConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = False
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     config.dataset_size = 10
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i in range(3):
    #     game = Mario()
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     config.dataset_size = 10
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # for i in range(3):
    # #     game = Mario()
    # #     config = cfg.MarioConfig()
    # #     config.seed = i
    # #     config.set_env(game)
    # #     config.bootstrap = 'smart'
    # #     config.div_loss = None
    # #     config.lambda_div = 10.0
    # #     config.use_diversity_sampling = True
    # #     config.generator_lr = 0.0001
    # #     config.discriminator_lr = 0.0001
    # #     config.discriminator_update_count = 5
    # #     config.use_spectral_norm_d = True
    # #     config.use_clipping_d = True
    # #     config.use_gradient_penalty = False
    # #     config.normalization_d = None
    # #     config.dataset_size = 10
    # #     models_dict = get_models(game, config)
    # #     trainer = Trainer(game, models_dict, config)
    # #     trainer.train()

    # for i in range(3):
    #     game = Mario()
    #     config = cfg.MarioConfig()
    #     config.set_env(game)
    #     config.seed = i
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1'
    #     config.lambda_div = 10.0
    #     config.use_diversity_sampling = True
    #     config.generator_lr = 0.0001
    #     config.discriminator_lr = 0.0001
    #     config.discriminator_update_count = 5
    #     config.use_spectral_norm_d = True
    #     config.use_clipping_d = True
    #     config.use_gradient_penalty = False
    #     config.normalization_d = None
    #     config.dataset_size = 10
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()
