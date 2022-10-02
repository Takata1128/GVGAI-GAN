import argparse
from gan.config import SmallModelConfig, SAModelConfig, DataExtendConfig
from gan.trainer import Trainer
from gan.level_dataset_extend import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    config = DataExtendConfig()
    config.set_env()
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.dataset_type = 'generated'
    config.bootstrap = "none"
    trainer = Trainer(config)
    trainer.train()

    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.use_minibatch_std = True
    #     config.bootstrap_interval = 20*config.train_batch_size
    #     config.dataset_size = 35
    #     config.bootstrap_max_count = 3
    #     trainer = Trainer(config)
    #     trainer.train()
    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.use_minibatch_std = True
    #     config.generator_filters = 64
    #     config.discriminator_filters = 64
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.use_minibatch_std = True
    #     config.div_loss = 'none'
    #     trainer = Trainer(config)
    #     trainer.train()
    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.use_minibatch_std = True
    #     config.bootstrap_filter = 1.0
    #     trainer = Trainer(config)
    #     trainer.train()
    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.use_minibatch_std = True
    #     config.div_loss = 'none'
    #     config.bootstrap_filter = 1.0
    #     trainer = Trainer(config)
    #     trainer.train()


# config = SmallModelConfig()
# config.set_env()
# config.seed = 0
# config.dataset_type = "train"
# if config.dataset_type == "train":
#     prepare_dataset(
#         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
#     )
# config.bootstrap_filter = 1.0
# config.div_loss = 'none'
# trainer = Trainer(config)
# trainer.train()

# config = SmallModelConfig()
# config.set_env()
# config.seed = 0
# config.dataset_type = "train"
# if config.dataset_type == "train":
#     prepare_dataset(
#         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
#     )
# config.bootstrap_filter = 1.0
# trainer = Trainer(config)
# trainer.train()

# config = SmallModelConfig()
# config.set_env()
# config.seed = 0
# config.dataset_type = "train"
# if config.dataset_type == "train":
#     prepare_dataset(
#         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
#     )
# config.div_loss = 'none'
# trainer = Trainer(config)
# trainer.train()
