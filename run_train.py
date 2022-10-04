import argparse
from gan.config import SmallModelConfig, SAModelConfig, DataExtendConfig
from gan.trainer import Trainer
from gan.level_dataset_extend import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    config = SmallModelConfig()
    config.env_name = "mario"
    config.set_env()
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    # config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.env_name = "mario"
    config.set_env()
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    # config = DataExtendConfig()
    # config.set_env()
    # config.dataset_type = "train"
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # trainer = Trainer(config)
    # trainer.train()

    # config = SmallModelConfig()
    # config.set_env()
    # config.dataset_type = 'generated'
    # config.bootstrap = "none"
    # trainer = Trainer(config)
    # trainer.train()
