import argparse
from gan.config import TrainingConfig
from gan.level_dataset_extend import prepare_dataset
from gan.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()
    config = TrainingConfig()
    # config.env_name = 'aliens'
    config.set_env()
    prepare_dataset(
        seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, clone_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    )
    trainer = Trainer(config)
    trainer.generator.summary()
    trainer.discriminator.summary()
