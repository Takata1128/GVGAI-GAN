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


class Experiments():
    def __init__(self, game):
        self.game = game

    def run(self, config: cfg.BaseConfig, exp_cnt=1):
        metrics = {}
        for i in range(exp_cnt):
            config.seed = i
            config.set_env(game)
            model_dict = get_models(game, config)
            trainer = Trainer(game, model_dict, config)
            m = trainer.train()
            for key, item in m.items():
                if key in metrics:
                    metrics[key] = item
                else:
                    metrics[key] += item
        for key, item in metrics.item():
            item /= exp_cnt
        return metrics


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


if __name__ == "__main__":
    for i in range(3):
        game = Boulderdash()
        config = cfg.BoulderdashConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = None
        config.div_loss = None
        config.use_diversity_sampling = False
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Boulderdash()
        config = cfg.BoulderdashConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = False
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Boulderdash()
        config = cfg.BoulderdashConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = True
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Boulderdash()
        config = cfg.BoulderdashConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'smart'
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Boulderdash()
        config = cfg.BoulderdashConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'smart'
        config.div_loss = None
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Zelda('v1')
        config = cfg.ZeldaConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = None
        config.div_loss = None
        config.use_diversity_sampling = False
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Zelda('v1')
        config = cfg.ZeldaConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = False
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Zelda('v1')
        config = cfg.ZeldaConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = True
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Zelda('v1')
        config = cfg.ZeldaConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'smart'
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = None
        config.div_loss = None
        config.use_diversity_sampling = False
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = False
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = None
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'smart'
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.bootstrap = None
        config.div_loss = None
        config.use_diversity_sampling = False
        config.dataset_size = 174
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = False
        config.dataset_size = 174
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.dataset_size = 174
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.set_env(game)
        config.seed = i
        config.bootstrap = 'smart'
        config.dataset_size = 174
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # lambda_ = [0.05, 0.1, 0.3]
    # for i, l in enumerate(lambda_):
    #     game = Boulderdash()
    #     config = cfg.BoulderdashConfig()
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1-latent'
    #     if l == 0:
    #         config.div_loss = None
    #     config.lambda_div = l
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i, l in enumerate(lambda_):
    #     game = Mario()
    #     config = cfg.MarioConfig()
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1-latent'
    #     if l == 0:
    #         config.div_loss = None
    #     config.lambda_div = l
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for i, l in enumerate(lambda_):
    #     game = Zelda('v1')
    #     config = cfg.ZeldaConfig()
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1-latent'
    #     if l == 0:
    #         config.div_loss = None
    #     config.lambda_div = l
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # filters = [0.85, 0.875, 0.90, 0.925, 0.95]
    # for f in filters:
    #     game = Mario()
    #     config = cfg.MarioConfig()
    #     config.set_env(game)
    #     config.div_loss = None
    #     config.bootstrap_hamming_filter = f
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for f in filters:
    #     game = Zelda('v1')
    #     config = cfg.ZeldaConfig()
    #     config.set_env(game)
    #     config.div_loss = None
    #     config.bootstrap_hamming_filter = f
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # for f in filters:
    #     game = Boulderdash()
    #     config = cfg.BoulderdashConfig()
    #     config.set_env(game)
    #     config.div_loss = None
    #     config.bootstrap_hamming_filter = f
    #     models_dict = get_models(game, config)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()
