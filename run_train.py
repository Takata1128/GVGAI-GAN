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

    div_loss = ['none', 'l1']
    generator_filters = [64, 128]
    discriminator_filters = [8, 16]

    for dl in div_loss:
        for gf in generator_filters:
            for df in discriminator_filters:
                config = SmallModelConfig()
                config.div_loss = dl
                config.generator_filters = gf
                config.discriminator_filters = df
                config.set_env()
                if config.dataset_type == "train":
                    prepare_dataset(
                        seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
                    )
                trainer = Trainer(config)
                trainer.train()
