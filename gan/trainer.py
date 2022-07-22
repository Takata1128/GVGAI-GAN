from __future__ import annotations
from ensurepip import bootstrap
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os
import torch
import operator
import numpy as np
from . import loss
from .level_visualizer import LevelVisualizer
from .dataset import LevelDataset
from .config import TrainingConfig
from .models import Generator, Discriminator
from .utils import (
    tensor_to_level_str,
    check_playable,
    check_level_similarity,
    check_object_similarity,
    check_shape_similarity,
)


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # reproducible settings
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

        if config.cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available else "cpu")
            print("device : cuda")
        else:
            self.device = torch.device("cpu")
            print("device : cpu")

        # Dataset
        train_dataset = LevelDataset(
            config.level_data_path,
            config.env_name,
            datamode="train",
            transform=None,
            latent_size=config.latent_size,
        )

        # dataset files
        self.dataset_files = os.listdir(train_dataset.image_dir)

        print("Training Dataset :", train_dataset.image_dir)

        # DataLoader
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True
        )

        # Level Visualizer
        self.level_visualizer = LevelVisualizer(
            config.env_name, config.env_version)

        # Network
        latent_shape = (config.latent_size,)
        self.generator = Generator(
            object_channel=7,
            shape_channel=2,
            shapes=config.model_shapes,
            z_shape=latent_shape,
            filters=config.generator_filters,
            is_self_attention=config.is_self_attention_g,
            is_conditional=config.is_conditional,
        ).to(self.device)
        self.discriminator = Discriminator(
            in_ch=config.input_shape[0],
            shapes=config.model_shapes[::-1],
            filters=config.discriminator_filters,
            is_self_attention=config.is_self_attention_d,
            is_minibatch_std=config.is_minibatch_std,
            is_spectral_norm=config.is_spectral_norm,
            is_conditional=config.is_conditional,
        ).to(self.device)

        # check model summary
        self.generator.summary(batch_size=self.config.train_batch_size)
        self.discriminator.summary(batch_size=self.config.train_batch_size)

        # Optimizer
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=config.generator_lr
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=config.discriminator_lr
        )

        self.playability = 0

    def train(self):
        wandb.login()
        metrics = {}
        max_playable_count = 0
        # summary(self.generator, (config.latent_size,), device=self.device)
        # summary(self.discriminator, config.input_shape, device=self.device)
        with wandb.init(project=f"{self.config.env_name} Level GAN", config=self.config.__dict__):
            step = 0
            latents_for_show = torch.randn(
                9,
                self.config.latent_size,
            ).to(self.device)
            latents_for_eval = torch.randn(
                self.config.eval_playable_counts,
                self.config.latent_size,
            ).to(self.device)
            labels_for_show = [
                [100, 86, 1, 1, 1, 1, 1, 1],
                [99, 87, 1, 1, 1, 1, 1, 1],
                [87, 99, 1, 1, 1, 1, 1, 1],
                [100, 86, 1, 1, 1, 1, 1, 1],
                [100, 86, 1, 1, 1, 1, 1, 1],
                [136, 50, 1, 1, 1, 1, 1, 1],
                [50, 136, 1, 1, 1, 1, 1, 1],
                [176, 10, 1, 1, 1, 1, 1, 1],
                [10, 176, 1, 1, 1, 1, 1, 1],
            ]
            labels_for_show = (
                torch.tensor(np.array(labels_for_show)).int().to(self.device)
            )
            labels_for_eval = []
            for i in range(self.config.eval_playable_counts // 5):
                labels_for_eval.append([100, 86, 1, 1, 1, 1, 1, 1])
                labels_for_eval.append([110, 76, 1, 1, 1, 1, 1, 1])
                labels_for_eval.append([120, 66, 1, 1, 1, 1, 1, 1])
                labels_for_eval.append([130, 56, 1, 1, 1, 1, 1, 1])
                labels_for_eval.append([140, 46, 1, 1, 1, 1, 1, 1])
            labels_for_eval = (
                torch.tensor(np.array(labels_for_eval)).int().to(self.device)
            )

            # model_save_path
            model_save_path = os.path.join(
                self.config.checkpoints_path,
                self.config.name + "-" + wandb.run.name.split("-")[-1],
            )
            os.makedirs(model_save_path)

            for epoch in range(1, self.config.epochs + 1):
                for latent, real, label in self.train_loader:
                    step += 1
                    latent, real, label = (
                        latent.to(self.device).float(),
                        real.to(self.device).float(),
                        label.to(self.device).int(),
                    )
                    fake_logit = self.generator(latent, label)

                    Dx, DGz_d, discriminator_loss = self._discriminator_update(
                        real, fake_logit, label
                    )
                    generator_loss, div_loss = self._generator_update(
                        fake_logit, label)

                    metrics["D(x)[Discriminator]"] = Dx
                    metrics["D(G(z))[Discriminator]"] = DGz_d
                    metrics["Discriminator Loss"] = discriminator_loss
                    metrics["Generator Loss"] = generator_loss
                    if self.config.div_loss != "none":
                        metrics["Generator Div Loss"] = div_loss
                    metrics["Epoch"] = epoch
                    wandb.log(metrics)

                    if step == self.config.steps:
                        break

                if epoch % self.config.save_image_interval_epoch == 0:
                    p_level = torch.nn.Softmax2d()(
                        self.generator(latents_for_show, labels_for_show))
                    level_strs = tensor_to_level_str(
                        self.config.env_name, p_level)
                    p_level_img = [
                        torch.Tensor(
                            np.array(self.level_visualizer.draw_level(lvl)).transpose(
                                2, 0, 1
                            )
                            / 255.0
                        )
                        for lvl in level_strs
                    ]

                    grid_level_img = make_grid(p_level_img, nrow=3, padding=0)
                    images = wandb.Image(
                        grid_level_img, caption="generated levels")
                    wandb.log({"Generated Levels": images})

                if epoch % self.config.eval_playable_interval_epoch == 0:
                    p_level = torch.nn.Softmax2d()(
                        self.generator(latents_for_eval, labels_for_eval))
                    level_strs = tensor_to_level_str(
                        self.config.env_name, p_level)
                    playable_levels = []
                    for level_str in level_strs:
                        if check_playable(level_str):
                            playable_levels.append(level_str)

                    if self.config.bootstrap and len(playable_levels) > 1:
                        self._bootstrap(playable_levels)

                    (
                        level_similarity,
                        object_similarity,
                        shape_similarity,
                        duplicate_ratio,
                    ) = self._calc_level_similarity(level_strs)

                    wandb.log(
                        {
                            "Playable Ratio": len(playable_levels)
                            / self.config.eval_playable_counts
                        }
                    )
                    wandb.log({"Level Similarity Ratio": level_similarity})
                    wandb.log({"Object Similarity Ratio": object_similarity})
                    wandb.log({"Shape Similarity Ratio": shape_similarity})
                    wandb.log({"Duplicate Ratio": duplicate_ratio})

                    if len(playable_levels) > max_playable_count:
                        self._save_models(
                            model_save_path, epoch, len(playable_levels))
                        max_playable_count = len(playable_levels)
                        self.playability = max_playable_count / self.config.eval_playable_counts

                if step == self.config.steps:
                    break

    # 生成データによる学習データ拡張 他との類似度が低いものを高いものと入れ替える
    def _bootstrap(self, playable_levels: list[str]):
        if self.config.bootstrap == "random":
            for level in playable_levels:
                self._level_expand(level)
        elif self.config.bootstrap == "smart":
            for i in range(len(playable_levels)):
                key = 0
                for j in range(len(playable_levels)):
                    if i == j:
                        continue
                    key += check_level_similarity(
                        playable_levels[i], playable_levels[j])
                playable_levels[i] = [playable_levels[i], key]

            playable_levels.sort(key=operator.itemgetter(1))

            exchange_count = min(
                self.config.dataset_max_change_count, len(playable_levels))

            change_levels = []
            for file in self.dataset_files:
                with open(
                    os.path.join(self.config.level_data_path,
                                 self.config.env_name, "train", file), mode="r"
                ) as f:
                    lvl_str = f.read()
                change_levels.append([file, lvl_str])

            for i in range(len(change_levels)):
                key = 0
                for j in range(len(change_levels)):
                    if i == j:
                        continue
                    key += check_level_similarity(
                        change_levels[i][1], change_levels[j][1])
                change_levels[i].append(key)

            change_levels.sort(
                key=operator.itemgetter(1), reverse=True)

            for i in range(exchange_count):
                file = change_levels[i][0]
                with open(
                    os.path.join(self.config.level_data_path,
                                 self.config.env_name, "train", file), mode="w"
                ) as f:
                    f.write(playable_levels[i][0])
        else:
            pass

    def _discriminator_update(self, real_images, fake_images, label=None):
        """
        update discriminator: maximize log(D(x)) + log(1-D(G(z)))
        """
        self.discriminator.zero_grad()
        real_logits = self.discriminator(real_images, label).view(-1)
        fake_logits = self.discriminator(
            torch.nn.Softmax2d()(fake_images).detach(), label).view(-1)
        if self.config.adv_loss == "baseline":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss(
                real_logits, fake_logits, self.config.label_flip_prob)
        elif self.config.adv_loss == "hinge":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss_hinge(
                real_logits, fake_logits
            )
        else:
            raise NotImplementedError()
        loss_real.backward()
        loss_fake.backward()
        discriminator_loss = loss_real + loss_fake
        self.optimizer_d.step()

        return D_x, D_G_z, discriminator_loss.item()

    def _generator_update(self, fake_images_logit, label=None):
        """
        update generator: maximize log(D(G(z)))
        """
        self.generator.zero_grad()
        fake_logits = self.discriminator(
            torch.nn.Softmax2d()(fake_images_logit), label).view(-1)
        if self.config.adv_loss == "baseline":
            generator_loss = loss.g_loss(fake_logits)
        elif self.config.adv_loss == "hinge":
            generator_loss = loss.g_loss_hinge(fake_logits)

        div_loss = loss.div_loss(
            torch.nn.Softmax2d()(fake_images_logit), self.config.div_loss, self.config.lambda_div)
        if self.playability > self.config.div_loss_threshold_playability:
            generator_loss += div_loss

        generator_loss.backward()
        self.optimizer_g.step()
        return generator_loss.item(), div_loss.item()

    def _save_models(self, model_save_path: str, epoch: int, playable_count: int):
        torch.save(
            {
                "epoch": epoch,
                "playable_count": playable_count,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            os.path.join(model_save_path, "models.tar"),
        )

    def _calc_level_similarity(self, level_strs):
        res_level, res_object, res_shape, res_duplicate = 0, 0, 0, 0
        n_level = 0
        n_object_level = 0
        n_shape_level = 0
        for i in range(0, len(level_strs)):
            for j in range(i + 1, len(level_strs)):
                res_level += check_level_similarity(
                    level_strs[i], level_strs[j])
                obj_tmp = check_object_similarity(level_strs[i], level_strs[j])
                if obj_tmp is not None:
                    res_object += obj_tmp
                    n_object_level += 1
                shape_tmp = check_shape_similarity(
                    level_strs[i], level_strs[j])
                if shape_tmp is not None:
                    res_shape += shape_tmp
                    n_shape_level += 1
                res_duplicate += (
                    1
                    if check_level_similarity(level_strs[i], level_strs[j]) >= 0.90
                    else 0
                )
                n_level += 1
        return (
            res_level / n_level,
            res_object / n_object_level if n_object_level > 0 else None,
            res_shape / n_shape_level if n_shape_level > 0 else None,
            res_duplicate / n_level,
        )

    def _level_expand(self, level_str):
        index = np.random.randint(0, len(self.dataset_files))
        file = self.dataset_files[index]
        with open(
            os.path.join(self.config.level_data_path,
                         self.config.env_name, "train", file),
            "w",
        ) as f:
            f.write(level_str)
