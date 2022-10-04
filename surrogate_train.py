from termios import tcdrain
import torch
import wandb
import sys
sys.path.append('../')
from gan.env import Env
from torch.utils.data import DataLoader
from surrogate.model import SurrogateModel
from surrogate.dataset import SurrogateModelDataset
import os
from gan.utils import tensor_to_level_str
from gan.config import SmallModelConfig
from gan.models.small_models import Generator


class SurrogateModelTrainingConfig():
    model_path: str = "/root/mnt/pcg/GVGAI-GAN/gan/checkpoints/none-357/models_2988.tar"
    level_path: str = "/root/mnt/pcg/GVGAI-GAN/surrogate/levels"
    train_datasize: int = 10000
    test_datasize: int = 256

    epochs: int = 300
    batch_size: int = 64
    eval_epoch_intervals: int = 10
    lr: float = 0.0001


def main(surrogate_config: SurrogateModelTrainingConfig, model_config: SmallModelConfig):
    if model_config.cuda:
        device = torch.device(
            f"cuda:{model_config.gpu_id}" if torch.cuda.is_available else "cpu")
        print(f"device : cuda:{model_config.gpu_id}")
    else:
        device = torch.device("cpu")
        print("device : cpu")

    env = Env(model_config.env_name, model_config.env_version)

    generator = Generator(
        out_dim=model_config.input_shape[0],
        shapes=model_config.model_shapes,
        z_shape=(model_config.latent_size,),
        filters=model_config.generator_filters,
        use_linear4z2features_g=model_config.use_linear4z2features_g,
        use_self_attention=model_config.use_self_attention_g,
        use_conditional=model_config.use_conditional,
        use_deconv_g=model_config.use_deconv_g
    ).to(device)

    model_dict = torch.load(surrogate_config.model_path)
    generator.load_state_dict(model_dict['generator'])

    surrogate_model = SurrogateModel(
        model_config.input_shape[0], model_config.model_shapes[::-1], filters=128, use_self_attention=False, use_bn=True).to(device)

    # ./levels以下にステージ生成
    def _generate_levels(latents, labels):
        p_level = torch.softmax(
            generator(latents, labels), dim=1)
        level_strs = tensor_to_level_str(model_config.env_name, p_level)
        return level_strs
    latents = torch.randn(
        surrogate_config.train_datasize,
        model_config.latent_size,
    ).to(device)
    levels = _generate_levels(latents, None)
    for i, level_tensor in enumerate(levels):
        with open(os.path.join(surrogate_config.level_path, model_config.env_name, 'train', f"level_{i}"), mode='w') as f:
            f.write(level_tensor)
    latents = torch.randn(
        surrogate_config.test_datasize,
        model_config.latent_size,
    ).to(device)
    levels = _generate_levels(latents, None)
    for i, level_tensor in enumerate(levels):
        with open(os.path.join(surrogate_config.level_path, model_config.env_name, 'test', f"level_{i}"), mode='w') as f:
            f.write(level_tensor)

    # データセット準備
    dataset = SurrogateModelDataset(
        surrogate_config.level_path, env, datamode='train', transform=None)
    train_loader = DataLoader(
        dataset, batch_size=surrogate_config.batch_size, shuffle=True)
    dataset = SurrogateModelDataset(
        surrogate_config.level_path, env, datamode='test', transform=None)
    test_loader = DataLoader(
        dataset, batch_size=surrogate_config.batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        surrogate_model.parameters(), lr=surrogate_config.lr)

    with wandb.init(project=f"Surrogate model training", config=surrogate_config.__dict__):
        step = 0
        for epoch in range(surrogate_config.epochs):
            generator.train()
            ac_count = 0
            metrics = {}
            for level_tensor, y in train_loader:
                step += 1
                level_tensor, y = (
                    level_tensor.to(device).float(),
                    torch.unsqueeze(y.to(device).float(), 1),
                )
                pred_y = surrogate_model(torch.softmax(level_tensor, dim=1))
                loss = criterion(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for py, ty in zip(pred_y, y):
                    if ty and py > 0.5:
                        ac_count += 1
                    elif not ty and py < 0.5:
                        ac_count += 1
                metrics["Loss"] = loss.item()
                wandb.log(metrics, step=step)
            metrics = {}
            metrics["Train Acc"] = ac_count / len(train_loader)
            metrics["Epochs"] = epoch
            wandb.log(metrics)

            if epoch % surrogate_config.eval_epoch_intervals == 0:
                generator.eval()
                with torch.no_grad():
                    ac_count = 0
                    for level_tensor, y in test_loader:
                        level_tensor, y = (
                            level_tensor.to(device).float(),
                            y.to(device).float(),
                        )
                        pred_y = surrogate_model(level_tensor)
                        for py, ty in zip(pred_y, y):
                            if ty and py > 0.5:
                                ac_count += 1
                            elif not ty and py < 0.5:
                                ac_count += 1
                            levels = _generate_levels(latents, None)
                            pred_y = surrogate_model(level_tensor)
                    metrics = {}
                    metrics["Test Acc"] = ac_count / len(test_loader)
                    wandb.log(metrics)


if __name__ == '__main__':
    config = SmallModelConfig()
    config.set_env()
    surrogate_config = SurrogateModelTrainingConfig()

    main(surrogate_config, config)
