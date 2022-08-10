import argparse
from operator import ge
from gan.config import NormalModelConfig, SmallModelConfig, TrainingConfig, DataExtendConfig, BranchModelConfig, OnlySAModelConfig
from gan.level_dataset_extend import prepare_dataset
from gan.small_models import Discriminator
from gan.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()
    # config = SmallModelConfig()
    # config.set_env()
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # trainer = Trainer(config)
    # trainer.train()

    dataset_types = ['generated_good', 'generated_part']
    batch_sizes = [32, 64]

    for ds in dataset_types:
        for bs in batch_sizes:
            config = SmallModelConfig()
            config.dataset_type = ds
            config.train_batch_size = bs
            config.set_env()
            if config.dataset_type == "train":
                prepare_dataset(
                    seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
                )
            trainer = Trainer(config)
            trainer.train()
