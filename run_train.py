import argparse
from gan.game.env import Game
from gan.game.zelda import Zelda
from gan.game.mario import Mario
from gan.game.boulderdash import Boulderdash
from gan.trainer import Trainer
from gan.level_dataset_extend import prepare_dataset
from gan.models.dcgan_models import DCGAN_G, DCGAN_D
from gan.models.general_models import Generator, Discriminator
import gan.config as cfg
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    def get_models(game: Game, config: cfg.BaseConfig):
        latent_shape = (config.latent_size,)
        device = torch.device(
            f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
        # generator = Generator(
        #     out_ch=game.input_shape[0],
        #     shapes=game.model_shape,
        #     z_shape=latent_shape,
        #     filters=config.generator_filters,
        #     use_linear4z2features_g=config.use_linear4z2features_g,
        #     use_self_attention=config.use_self_attention_g,
        #     use_conditional=config.use_conditional,
        #     use_deconv_g=config.use_deconv_g
        # ).to(device)
        # discriminator = Discriminator(
        #     in_ch=game.input_shape[0],
        #     shapes=game.model_shape[::-1],
        #     filters=config.discriminator_filters,
        #     use_bn=config.use_bn_d,
        #     use_self_attention=config.use_self_attention_d,
        #     use_minibatch_std=config.use_minibatch_std,
        #     use_recon_loss=config.use_recon_loss,
        #     use_conditional=config.use_conditional,
        #     use_sn=config.use_sn_d,
        #     use_pooling=config.use_pooling_d
        generator = Generator(
            isize=game.input_shape[1], nz=config.latent_size, nc=game.input_shape[
                0], ngf=config.generator_filters, self_attention=config.use_self_attention_g
        ).to(device)
        discriminator = Discriminator(
            isize=game.input_shape[1], nz=config.latent_size, nc=game.input_shape[
                0], ndf=config.discriminator_filters, use_self_attention=config.use_self_attention_d, use_spectral_norm=config.use_spectral_norm_d
        )
        models_dict = {
            'generator': generator,
            'discriminator': discriminator
        }
        return models_dict

    # game = Zelda('zelda', 'v1')
    # config = cfg.ZeldaConfig()
    # config.set_env(game)
    # config.adv_loss = 'hinge'
    # config.discrimianator_update_count = 1
    # config.generator_lr = 0.0001
    # config.discriminator_lr = 0.0004
    # config.bootstrap = 'baseline'
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Mario('mario', 'v0')
    # config = cfg.MarioConfig()
    # config.set_env(game)
    # config.bootstrap = None
    # config.div_loss = None
    # config.use_self_attention_d = False
    # config.use_self_attention_g = False
    # config.use_diversity_sampling = False
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Mario('mario', 'v0')
    # config = cfg.MarioConfig()
    # config.set_env(game)
    # config.generator_lr = 0.00005
    # config.discriminator_lr = 0.00005
    # config.bootstrap = None
    # config.div_loss = None
    # config.use_self_attention_d = False
    # config.use_self_attention_g = False
    # config.use_diversity_sampling = False
    # config.dataset_size = 10
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Boulderdash('boulderdash', 'v0')
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = 'baseline'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.discriminator_lr = 0.0001
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Boulderdash('boulderdash', 'v0')
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = 'baseline'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.discriminator_lr = 0.0001
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = 'l1'
    # config.lambda_div = 0.1
    # config.use_diversity_sampling = False
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Boulderdash('boulderdash', 'v0')
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = 'smart'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.discriminator_lr = 0.0001
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = None
    # config.use_diversity_sampling = False
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Boulderdash('boulderdash', 'v0')
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = 'baseline'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.discriminator_lr = 0.0001
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Boulderdash('boulderdash', 'v0')
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = 'smart'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.discriminator_lr = 0.0001
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = None
    # config.use_diversity_sampling = True
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # game = Boulderdash('boulderdash', 'v0')
    # config = cfg.BoulderdashConfig()
    # config.set_env(game)
    # config.bootstrap = 'smart'
    # config.adv_loss = 'wgan'
    # config.discrimianator_update_count = 5
    # config.discriminator_lr = 0.0001
    # config.use_self_attention_g = False
    # config.use_self_attention_d = False
    # config.div_loss = 'l1'
    # config.lambda_div = 0.1
    # config.use_diversity_sampling = True
    # prepare_dataset(
    #     game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # models_dict = get_models(game, config, device)
    # trainer = Trainer(game, models_dict, config)
    # trainer.train()

    # # DCGAN
    # # conventional bootstrap
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.adv_loss = 'wgan'
    #     config.lambda_div = 0.5
    #     config.discrimianator_update_count = 5
    #     config.discriminator_lr = 0.0001
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # L1
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.adv_loss = 'wgan'
    #     config.lambda_div = 0.5
    #     config.discrimianator_update_count = 5
    #     config.discriminator_lr = 0.0001
    #     config.bootstrap = 'baseline'
    #     config.div_loss = 'l1'
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # hamming filter
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.adv_loss = 'wgan'
    #     config.lambda_div = 0.5
    #     config.discrimianator_update_count = 5
    #     config.discriminator_lr = 0.0001
    #     config.bootstrap = 'smart'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # diversity sampling
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.adv_loss = 'wgan'
    #     config.lambda_div = 0.5
    #     config.discrimianator_update_count = 5
    #     config.discriminator_lr = 0.0001
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = True
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # ours
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.adv_loss = 'wgan'
    #     config.lambda_div = 0.5
    #     config.discrimianator_update_count = 5
    #     config.discriminator_lr = 0.0001
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1'
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = True
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # DCGAN
    # conventional bootstrap
    # for i in range(1):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.adv_loss = 'wgan'
    #     config.div_loss = None
    #     config.lambda_div = 0.5
    #     config.discrimianator_update_count = 5
    #     config.discriminator_lr = 0.0001
    #     config.bootstrap = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     config.dataset_size = 10
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # DCGAN
    # conventional bootstrap
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.adv_loss = 'wgan'
        config.discrimianator_update_count = 5
        config.discriminator_lr = 0.0001
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = False
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # L1
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.adv_loss = 'wgan'
        config.discrimianator_update_count = 5
        config.discriminator_lr = 0.0001
        config.bootstrap = 'baseline'
        config.div_loss = 'l1'
        config.lambda_div = 0.75
        config.use_diversity_sampling = False
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # hamming filter
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.adv_loss = 'wgan'
        config.discrimianator_update_count = 5
        config.discriminator_lr = 0.0001
        config.bootstrap = 'smart'
        config.div_loss = None
        config.use_diversity_sampling = False
        config.bootstrap_hamming_filter = 0.95
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # diversity sampling
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.adv_loss = 'wgan'
        config.discrimianator_update_count = 5
        config.discriminator_lr = 0.0001
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = True
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # ours
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.adv_loss = 'wgan'
        config.discrimianator_update_count = 5
        config.discriminator_lr = 0.0001
        config.bootstrap = 'smart'
        config.div_loss = 'l1'
        config.lambda_div = 0.75
        config.use_diversity_sampling = True
        config.bootstrap_hamming_filter = 0.95
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # ours
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.adv_loss = 'wgan'
        config.discrimianator_update_count = 5
        config.discriminator_lr = 0.0001
        config.bootstrap = 'smart'
        config.div_loss = None
        config.use_diversity_sampling = True
        config.bootstrap_hamming_filter = 0.95
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # DCGAN
    # conventional bootstrap
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = False
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # L1
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = 'l1'
        config.use_diversity_sampling = False
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # hamming filter
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'smart'
        config.div_loss = None
        config.use_diversity_sampling = False
        config.bootstrap_hamming_filter = 0.95
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # diversity sampling
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'baseline'
        config.div_loss = None
        config.use_diversity_sampling = True
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # ours
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'smart'
        config.div_loss = 'l1'
        config.use_diversity_sampling = True
        config.bootstrap_hamming_filter = 0.95
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # ours
    for i in range(3):
        game = Mario()
        config = cfg.MarioConfig()
        config.seed = i
        config.set_env(game)
        config.bootstrap = 'smart'
        config.div_loss = None
        config.use_diversity_sampling = True
        config.bootstrap_hamming_filter = 0.95
        config.dataset_size = 10
        config.bootstrap_epoch = 10 * (i + 1)
        models_dict = get_models(game, config)
        trainer = Trainer(game, models_dict, config)
        trainer.train()

    # # DCGAN
    # # conventional bootstrap
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # L1
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = 'l1'
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # hamming filter
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # diversity sampling
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = True
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # ours
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1'
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = True
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # DCGAN
    # # conventional bootstrap
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     config.dataset_size = 10
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # L1
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = 'l1'
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     config.dataset_size = 10
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # hamming filter
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = False
    #     config.dataset_size = 10
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # diversity sampling
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'baseline'
    #     config.div_loss = None
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = True
    #     config.dataset_size = 10
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()

    # # ours
    # for i in range(3):
    #     game = Mario('mario', 'v0')
    #     config = cfg.MarioConfig()
    #     config.seed = i
    #     config.set_env(game)
    #     config.bootstrap = 'smart'
    #     config.div_loss = 'l1'
    #     config.use_self_attention_d = False
    #     config.use_self_attention_g = False
    #     config.use_diversity_sampling = True
    #     config.dataset_size = 10
    #     prepare_dataset(
    #         game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
    #     device = torch.device(
    #         f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #     models_dict = get_models(game, config, device)
    #     trainer = Trainer(game, models_dict, config)
    #     trainer.train()
