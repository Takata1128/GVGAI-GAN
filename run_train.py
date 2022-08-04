import argparse
from gan.config import NormalModelConfig, SmallModelConfig, TrainingConfig, DataExtendConfig, BranchModelConfig
from gan.level_dataset_extend import prepare_dataset
from gan.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()
    config = DataExtendConfig()
    config.set_env()
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    trainer = Trainer(config)
    trainer.train()
