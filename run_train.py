import argparse
from gan.config import SmallModelConfig, SAModelConfig, DataExtendConfig
from gan.trainer import Trainer
from gan.level_dataset_extend import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()

    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.set_env()
    #     config.seed = i
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.eval_playable_interval = 20 * config.train_batch_size
    #     config.save_image_interval = 20 * config.train_batch_size
    #     config.save_model_interval = 100 * config.train_batch_size
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.set_env()
    #     config.seed = i
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.bootstrap_filter = 1.0
    #     config.div_loss = 'none'
    #     config.eval_playable_interval = 20 * config.train_batch_size
    #     config.save_image_interval = 20 * config.train_batch_size
    #     config.save_model_interval = 100 * config.train_batch_size
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(3):
    #     config = SmallModelConfig()
    #     config.set_env()
    #     config.seed = i
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.use_recon_loss = False
    #     config.eval_playable_interval = 20 * config.train_batch_size
    #     config.save_image_interval = 20 * config.train_batch_size
    #     config.save_model_interval = 100 * config.train_batch_size
    #     trainer = Trainer(config)
    #     trainer.train()

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
    # config.dataset_type = "generated_0913"
    # config.bootstrap = 'none'
    # config.steps = 10000
    # trainer = Trainer(config)
    # trainer.train()
    config = SmallModelConfig()
    config.set_env()
    config.seed = 0
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 0
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.bootstrap_filter = 1.0
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 0
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.bootstrap_filter = 1.0
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 0
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 1
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 1
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.bootstrap_filter = 1.0
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 1
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.bootstrap_filter = 1.0
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 1
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 2
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 2
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.bootstrap_filter = 1.0
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 2
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.bootstrap_filter = 1.0
    trainer = Trainer(config)
    trainer.train()

    config = SmallModelConfig()
    config.set_env()
    config.seed = 2
    config.dataset_type = "train"
    if config.dataset_type == "train":
        prepare_dataset(
            seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        )
    config.div_loss = 'none'
    trainer = Trainer(config)
    trainer.train()

    # config = SmallModelConfig()
    # config.set_env()
    # config.dataset_type = "train"
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.bootstrap = "smart"
    # config.div_loss = 'l1'
    # config.use_self_attention_d = False
    # config.use_self_attention_g = False
    # config.eval_playable_interval = 20 * config.train_batch_size
    # config.save_image_interval = 20 * config.train_batch_size
    # config.save_model_interval = 100 * config.train_batch_size
    # trainer = Trainer(config)
    # trainer.train()

    # config = SmallModelConfig()
    # config.set_env()
    # config.dataset_type = "train"
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.bootstrap = "smart"
    # config.div_loss = 'none'
    # config.bootstrap_filter = 1.0
    # config.eval_playable_interval = 20 * config.train_batch_size
    # config.save_image_interval = 20 * config.train_batch_size
    # config.save_model_interval = 100 * config.train_batch_size
    # config.bootstrap_filter = 1.0
    # trainer = Trainer(config)
    # trainer.train()

    # lambda_divs = [1.0, 5.0, 10.0, 50.0]
    # for div in lambda_divs:
    #     config = SmallModelConfig()
    #     config.set_env()
    #     config.dataset_type = "train"
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     config.bootstrap = "smart"
    #     config.div_loss = 'l2'
    #     config.eval_playable_interval = 20 * config.train_batch_size
    #     config.save_image_interval = 20 * config.train_batch_size
    #     config.save_model_interval = 100 * config.train_batch_size
    #     config.lambda_div = div
    #     trainer = Trainer(config)
    #     trainer.train()

    # config = SmallModelConfig()
    # config.set_env()
    # config.dataset_type = "generated_fixed"
    # config.eval_playable_interval = 20 * config.train_batch_size
    # config.save_image_interval = 20 * config.train_batch_size
    # # config.use_minibatch_std = False
    # # config.use_sn_d = True
    # trainer = Trainer(config)
    # trainer.train()
