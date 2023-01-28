import os
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
import matplotlib.pyplot as plt

RESULT_PATH = '/root/mnt/pcg/GVGAI-GAN/experimental_results'


class Experiments():
    def __init__(self, game):
        self.game = game

    def run(self, config: cfg.BaseConfig, exp_cnt=1):
        metrics = {}
        metrics['wandb'] = {}
        metrics['other'] = {}
        metrics['other'][r'X% Duplication Rate'] = [[], []]
        for i in range(exp_cnt):
            config.seed = i
            config.set_env(self.game)
            model_dict = get_models(self.game, config)
            trainer = Trainer(self.game, model_dict, config)
            m = trainer.train()
            for key, item in m['wandb'].items():
                if key not in metrics['wandb']:
                    metrics['wandb'][key] = item
                else:
                    metrics['wandb'][key] += item
            if 'other' in m:
                if i == 0:
                    metrics['other'][r'X% Duplication Rate'][0] = m['other'][r'X% Duplication Rate'][0]
                    metrics['other'][r'X% Duplication Rate'][1] = m['other'][r'X% Duplication Rate'][1]
                else:
                    for j in range(len(metrics['other'][r'X% Duplication Rate'][1])):
                        metrics['other'][r'X% Duplication Rate'][1][j] += m['other'][r'X% Duplication Rate'][1][j]

        for key, item in metrics['wandb'].items():
            metrics['wandb'][key] /= exp_cnt
        for j in range(len(metrics['other'][r'X% Duplication Rate'][1])):
            metrics['other'][r'X% Duplication Rate'][1][j] /= exp_cnt
        return metrics


def report(results: list, legends=[], name: str = None):
    '''
    results -
        Final Playable Rate
        Final Duplication Rate
        Similarity Rate
        Features Type Nums
        Features Duplication Rate    
    '''
    with open(os.path.join(RESULT_PATH, name), 'w') as f:
        legend_str = ''
        for key, item in results[0]['wandb'].items():
            legend_str += str(key)
            legend_str += ','
        f.write(legend_str + '\n')
    plt.clf()
    with open(os.path.join(RESULT_PATH, name), 'a') as f:
        for i, res in enumerate(results):
            data_str = ''
            for key, item in res['wandb'].items():
                data_str += str(item)
                data_str += ','
            if 'other' in res and r'X% Duplication Rate' in res['other']:
                x, y = res['other'][r'X% Duplication Rate']
                plt.plot(x, y, label=legends[i]
                         if i < len(legends) else str(i))
            f.write(data_str + '\n')
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, name) + '_dpl.png', format='png')


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
    # zelda_exp = Experiments(Zelda('v1'))
    # config = cfg.ZeldaConfig()
    # results = []

    # config.bootstrap = None
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # result = zelda_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'baseline'
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # result = zelda_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'baseline'
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # result = zelda_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'smart'
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # result = zelda_exp.run(config, 3)
    # results.append(result)

    # report(results, ['simple', 'bootstrapping', 'diversity sampling', 'ours'], 'zelda2')

    # boulderdash_exp = Experiments(Boulderdash())
    # config = cfg.BoulderdashConfig()
    # results = []

    # config.bootstrap = None
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # result = boulderdash_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'baseline'
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # result = boulderdash_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'baseline'
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # result = boulderdash_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'smart'
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # result = boulderdash_exp.run(config, 3)
    # results.append(result)

    # report(results, ['simple', 'bootstrapping',
    #        'diversity sampling', 'ours'], 'boulderdash')

    # mario_exp = Experiments(Mario())
    # config = cfg.MarioConfig()
    # results = []

    # config.bootstrap = None
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # result = mario_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'baseline'
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # result = mario_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'baseline'
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # result = mario_exp.run(config, 3)
    # results.append(result)

    # config.bootstrap = 'smart'
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # result = mario_exp.run(config, 3)
    # results.append(result)

    # report(results, ['simple', 'bootstrapping',
    #        'diversity sampling', 'ours'], 'mario')

    lambdas = [0, 0.05, 0.1, 0.3]

    zelda_exp = Experiments(Zelda('v1'))
    config = cfg.ZeldaConfig()
    results = []
    for l in lambdas:
        config.bootstrap = 'smart'
        config.div_loss = 'l1-latent'
        config.lambda_div = l
        config.use_diversity_sampling = True
        result = zelda_exp.run(config, 1)
        results.append(result)
    report(results, ['λ=0', 'λ=0.05', 'λ=0.1', 'λ=0.3'], 'zelda_divloss_new')

    boulderdash_exp = Experiments(Boulderdash())
    config = cfg.BoulderdashConfig()
    results = []
    for l in lambdas:
        config.bootstrap = 'smart'
        config.div_loss = 'l1-latent'
        config.lambda_div = l
        config.use_diversity_sampling = True
        result = boulderdash_exp.run(config, 1)
        results.append(result)
    report(results, ['λ=0', 'λ=0.05', 'λ=0.1', 'λ=0.3'],
           'boulderdash_divloss_new')

    mario_exp = Experiments(Mario())
    config = cfg.MarioConfig()
    results = []
    for l in lambdas:
        config.bootstrap = 'smart'
        config.lambda_div = l
        config.use_diversity_sampling = True
        result = mario_exp.run(config, 1)
        results.append(result)
    report(results, ['λ=0', 'λ=0.05', 'λ=0.1', 'λ=0.3'], 'mario_divloss_new')

    filters = [0.80, 0.85, 0.90, 0.95, 1.0]

    zelda_exp = Experiments(Zelda('v1'))
    config = cfg.ZeldaConfig()
    results = []
    for f in filters:
        config.bootstrap = 'smart'
        config.div_loss = None
        config.bootstrap_hamming_filter = f
        config.use_diversity_sampling = True
        result = zelda_exp.run(config, 1)
        results.append(result)
    report(results, ['80.0', '85.0', '90.0',
           '95.0', '100.0'], 'zelda_filter_new')

    boulderdash_exp = Experiments(Boulderdash())
    config = cfg.BoulderdashConfig()
    results = []
    for f in filters:
        config.bootstrap = 'smart'
        config.div_loss = None
        config.bootstrap_hamming_filter = f
        config.use_diversity_sampling = True
        result = boulderdash_exp.run(config, 1)
        results.append(result)
    report(results, ['80.0', '85.0', '90.0',
           '95.0', '100.0'], 'boulderdash_filter_new')

    mario_exp = Experiments(Mario())
    config = cfg.MarioConfig()
    results = []
    for f in filters:
        config.bootstrap = 'smart'
        config.div_loss = None
        config.bootstrap_hamming_filter = f
        config.use_diversity_sampling = True
        result = mario_exp.run(config, 1)
        results.append(result)
    report(results, ['80.0', '85.0', '90.0',
           '95.0', '100.0'], 'mario_filter_new')
