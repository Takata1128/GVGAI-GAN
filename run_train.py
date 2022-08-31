import argparse
from gan.config import NormalModelConfig, SmallModelConfig, TrainingConfig, DataExtendConfig, BranchModelConfig, OnlySAModelConfig
from gan.level_dataset_extend import prepare_dataset
from gan.small_models import Discriminator
from gan.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()
    # config = DataExtendConfig()
    # config.set_env()
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # trainer = Trainer(config)
    # trainer.train()

    for i in range(5):
        config = SmallModelConfig()
        config.set_env()
        config.seed = i
        config.use_sn_d = True
        trainer = Trainer(config)
        trainer.train()

    for i in range(5):
        config = SmallModelConfig()
        config.set_env()
        config.seed = i
        config.adv_loss = 'baseline'
        config.smooth_label_value = 0.9
        trainer = Trainer(config)
        trainer.train()

    # for i in range(5):
    #     config = SmallModelConfig()
    #     config.set_env()
    #     config.seed = i
    #     config.discriminator_filters = 128
    #     trainer = Trainer(config)
    #     trainer.train()

    # latent_sizes = [32, 64]
    # filters = [64, 128]
    # self_atention = [True, False]

    # for ls in latent_sizes:
    #     for filter in filters:
    #         for sa in self_atention:
    #             config = SmallModelConfig()
    #             config.latent_sizes = ls
    #             config.generator_filters = filter
    #             config.discriminator_filters = filter
    #             config.use_self_attention_d = sa
    #             config.use_self_attention_g = sa
    #             config.set_env()
    #             trainer = Trainer(config)
    #             trainer.train()

    # use_minibatch_std = [False, True]
    # use_conditional = [False, True]
    # use_recon_loss = [False, True]
    # adv_losses = ["baseline", "hinge"]

    # for mstd in use_minibatch_std:
    #     for uc in use_conditional:
    #         for rec in use_recon_loss:
    #             for loss in adv_losses:
    #                 config = SmallModelConfig()
    #                 config.use_minibatch_std = mstd
    #                 config.use_recon_loss = rec
    #                 config.use_conditional = uc
    #                 config.adv_loss = loss
    #                 config.set_env()
    #                 trainer = Trainer(config)
    #                 trainer.train()
