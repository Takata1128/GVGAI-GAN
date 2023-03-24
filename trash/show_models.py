import argparse
from gan.game.zelda import Zelda
from gan.game.mario import Mario
from gan.trainer import Trainer
from gan.prepare_dataset import prepare_dataset
from gan.models.small_models import Generator, Discriminator
# from gan.models.general_models import Generator, Discriminator
from gan.models.dcgan_models import DCGAN_G, DCGAN_D
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
        # generator = Generator(
        #     isize=16, nz=config.latent_size, nc=game.input_shape[0], ngf=config.generator_filters, self_attention=config.use_self_attention_g
        # ).to(device)
        # discriminator = Discriminator(
        #     isize=16, nz=config.latent_size, nc=game.input_shape[0], ndf=config.discriminator_filters, self_attention=config.use_self_attention_d
        # )
        models_dict = {
            'generator': generator,
            'discriminator': discriminator
        }
        return models_dict

    game = Zelda('zelda', 'v1')
    config = cfg.ZeldaConfig()
    config.set_env(game)
    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models_dict = get_models(game, config, device)
    models_dict['generator'].summary()
    models_dict['discriminator'].summary()

    # config = cfg.MarioConfig()
    # config.env_name = "mario"
    # config.env_version = 'v0'
    # game = Mario(config.env_name, config.env_version)
    # config.set_env()
    # device = torch.device(
    #     f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # generator = DCGAN_G(32, config.latent_size,
    #                     game.input_shape[0], config.generator_filters)
    # discriminator = DCGAN_D(32, config.latent_size,
    #                         game.input_shape[0], config.discriminator_filters)
    # models_dict = {
    #     'generator': generator,
    #     'discriminator': discriminator
    # }
    # models_dict['generator'].summary()
    # models_dict['discriminator'].summary()
