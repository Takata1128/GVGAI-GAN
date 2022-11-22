import argparse
from ensurepip import bootstrap
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
    # config.env_name = 'roguelike'
    # config.env_version = 'v0'
    # config.set_env()
    # config.dataset_type = 'train'
    # config.dataset_size = 300
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.bootstrap_max_count = 10
    # config.save_image_epoch: int = 10
    # config.save_model_epoch: int = 100
    # config.eval_epoch: int = 10
    # config.bootstrap_epoch: int = 10
    # config.div_loss = 'none'
    # config.bootstrap = 'baseline'
    # config.eval_playable_counts = 100
    # # config.use_self_attention_g = []
    # # config.use_self_attention_d = []
    # config.use_conditional = True
    # config.use_recon_loss = False
    # trainer = Trainer(config)
    # trainer.train()

    # ours
    # config = SmallModelConfig()
    # config.seed = 1009
    # config.env_name = 'roguelike'
    # config.env_version = 'v0'
    # config.set_env()
    # config.dataset_type = 'train'
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # trainer = Trainer(config)
    # trainer.train()

    # config = SmallModelConfig()
    # config.env_name = "mario"
    # config.env_version = 'v0'
    # config.set_env()
    # config.dataset_size = 174
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.use_self_attention_g = []
    # config.use_self_attention_d = []
    # config.bootstrap = None
    # config.div_loss = None
    # trainer = Trainer(config)
    # trainer.train()

    # for i in range(5):
    #     # baseline
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'train'
    #     config.dataset_size = 35
    #     config.bootstrap_epoch = 1
    #     config.div_loss = None
    #     config.bootstrap = 'baseline'
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    for i in range(1):
        # ours
        config = SmallModelConfig()
        config.seed = i
        config.bootstrap_property_filter = None
        config.set_env()
        config.dataset_type = 'train'
        config.dataset_size = 35
        config.bootstrap_epoch = 10
        if config.dataset_type == "train":
            prepare_dataset(
                seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
            )
        trainer = Trainer(config)
        trainer.train()

    for i in range(1):
        # ours2
        # config = DataExtendConfig()
        # config.seed = i
        # config.set_env()
        # config.dataset_type = "train"
        # if config.dataset_type == "train":
        #     prepare_dataset(
        #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
        #     )
        # trainer = Trainer(config)
        # trainer.train()

        config = SmallModelConfig()
        config.name = 'SAmodel'
        config.seed = i
        config.save_image_epoch = 20
        config.set_env()
        config.dataset_type = 'generated'
        config.save_model_interval = 200 * config.train_batch_size
        config.bootstrap = None
        config.div_loss = None
        trainer = Trainer(config)
        trainer.train()

    # only bootstrap
    # config = SmallModelConfig()
    # config.seed = 1009
    # config.set_env()
    # config.dataset_type = 'train'
    # if config.dataset_type == "train":
    #     prepare_dataset(
    #         seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #     )
    # config.div_loss = 'none'
    # config.bootstrap_filter = 1.0
    # config.bootstrap = 'baseline'
    # config.use_conditional = True
    # trainer = Trainer(config)
    # trainer.train()

    # bootstrap_epochs = [10]

    # for be in bootstrap_epochs:
    # for i in range(5):
    #     # simple GAN
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'train'
    #     config.dataset_size = 1000
    #     config.div_loss = None
    #     config.bootstrap = None
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(5):
    #     # baseline
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'train'
    #     config.dataset_size = 35
    #     config.div_loss = None
    #     config.bootstrap = 'baseline'
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(5):
    #     # ours
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'train'
    #     config.dataset_size = 35
    #     config.div_loss = 'l1'
    #     config.lambda_div = 50.0
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(5):
    #     # ours2
    #     config = DataExtendConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     config.div_loss = 'l1'
    #     config.lambda_div = 50.0
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'generated'
    #     config.save_model_interval = 200 * config.train_batch_size
    #     config.bootstrap = None
    #     config.div_loss = None
    #     config.use_conditional = False
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(5):
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'train'
    #     config.div_loss = None
    #     config.bootstrap = 'baseline'
    #     config.dataset_size = 35
    #     config.bootstrap_epoch = 20
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(5):
    #     # ours
    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'train'
    #     config.div_loss = 'l1'
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    # for i in range(5):
    #     # ours2
    #     config = DataExtendConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = "train"
    #     config.div_loss = 'l1'
    #     config.lambda_div = 50.0
    #     config.use_conditional = False
    #     if config.dataset_type == "train":
    #         prepare_dataset(
    #             seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size, game_name=config.env_name, version=config.env_version
    #         )
    #     trainer = Trainer(config)
    #     trainer.train()

    #     config = SmallModelConfig()
    #     config.seed = i
    #     config.set_env()
    #     config.dataset_type = 'generated'
    #     config.save_model_interval = 200 * config.train_batch_size
    #     config.bootstrap = None
    #     config.div_loss = None
    #     config.use_conditional = True
    #     trainer = Trainer(config)
    #     trainer.train()

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
