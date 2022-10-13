import argparse
from gan.config import SmallModelConfig, DataExtendConfig
from gan.trainer import Trainer
from gan.level_dataset_extend import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    # baseline
    # config = SmallModelConfig()
    # config.seed = 1009
    # config.set_env()
    # config.dataset_type = 'train'
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.div_loss = 'none'
    # config.bootstrap = 'none'
    # config.use_conditional = True
    # config.use_recon_loss = False
    # trainer = Trainer(config)
    # trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.dataset_type = 'generated'
    config.save_model_interval = 200 * config.train_batch_size
    config.bootstrap = "none"
    config.div_loss = 'none'
    # config.use_conditional = True
    config.use_recon_loss = False
    trainer = Trainer(config)
    trainer.train()

    # only bootstrap
    config = SmallModelConfig()
    config.seed = 1009
    config.set_env()
    config.dataset_type = 'train'
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.div_loss = 'none'
    config.bootstrap_filter = 1.0
    config.use_conditional = True
    config.use_recon_loss = False
    trainer = Trainer(config)
    trainer.train()

    # ours
    config = SmallModelConfig()
    config.seed = 1009
    config.set_env()
    config.dataset_type = 'train'
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.use_conditional = True
    config.use_recon_loss = False
    trainer = Trainer(config)
    trainer.train()

    # ours2
    config = DataExtendConfig()
    config.seed = 1009
    config.set_env()
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.use_conditional = True
    config.use_recon_loss = False
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.dataset_type = 'generated'
    config.save_model_interval = 200 * config.train_batch_size
    config.bootstrap = "none"
    config.div_loss = 'none'
    config.use_conditional = True
    config.use_recon_loss = False
    trainer = Trainer(config)
    trainer.train()

    # config = SmallModelConfig()
    # config.env_name = "mario"
    # config.set_env()
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # # config.div_loss = 'none'
    # trainer = Trainer(config)
    # trainer.train()

    # config = SmallModelConfig()
    # config.env_name = "mario"
    # config.set_env()
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.div_loss = 'none'
    # trainer = Trainer(config)
    # trainer.train()
