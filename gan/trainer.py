from __future__ import annotations
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os
import torch
import operator
import numpy as np
import random


from . import loss
from .env import Env
from .level_visualizer import LevelVisualizer
from .dataset import LevelDataset
from .config import DataExtendConfig, TrainingConfig
from .utils import (
    tensor_to_level_str,
    check_playable,
    check_level_similarity,
    check_object_similarity,
    check_shape_similarity,
)
from gan import dataset


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = Env(self.config.env_name, self.config.env_version)

        # reproducible settings
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if config.cuda:
            self.device = torch.device(
                f"cuda:{config.gpu_id}" if torch.cuda.is_available else "cpu")
            print(f"device : cuda:{config.gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("device : cpu")

        # Dataset
        train_dataset = LevelDataset(
            config.level_data_path,
            self.env,
            datamode=self.config.dataset_type,
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
        self.level_visualizer = LevelVisualizer(self.env)

        # Network
        self._build_model()

        # Other
        self.playability = 0
        self.data_index = 0
        self.steps_per_epoch = len(self.train_loader)
        self.model_save_epoch = self.config.save_model_interval//(
            self.config.train_batch_size*self.steps_per_epoch)
        self.image_save_epoch = self.config.save_image_interval//(
            self.config.train_batch_size*self.steps_per_epoch)
        self.eval_epoch = self.config.eval_playable_interval//(
            self.config.train_batch_size*self.steps_per_epoch)
        self.bootstrap_epoch = self.config.bootstrap_interval//(
            self.config.train_batch_size*self.steps_per_epoch)
        print(f"model_save_epoch:{self.model_save_epoch}")
        print(f"image_save_epoch:{self.image_save_epoch}")
        print(f"eval_epoch:{self.eval_epoch}")
        print(f"bootstrap_epoch:{self.bootstrap_epoch}")

        if isinstance(self.config, DataExtendConfig):
            self.reset_epoch = self.config.reset_weight_interval//(
                self.config.train_batch_size*self.steps_per_epoch)
            print(f"model_reset_epoch:{self.reset_epoch}")

    def _build_model(self):
        latent_shape = (self.config.latent_size,)

        if self.config.model_type == "normal":
            from .models.models import Generator
            self.generator = Generator(
                out_dim=self.config.input_shape[0],
                shapes=self.config.model_shapes,
                z_shape=latent_shape,
                filters=self.config.generator_filters,
                is_self_attention=self.config.use_self_attention_g,
                is_conditional=self.config.use_conditional,
            ).to(self.device)
        elif self.config.model_type == "small":
            from .models.small_models import Generator
            self.generator = Generator(
                out_dim=self.config.input_shape[0],
                shapes=self.config.model_shapes,
                z_shape=latent_shape,
                filters=self.config.generator_filters,
                use_linear4z2features_g=self.config.use_linear4z2features_g,
                use_self_attention=self.config.use_self_attention_g,
                use_conditional=self.config.use_conditional,
                use_deconv_g=self.config.use_deconv_g
            ).to(self.device)
        elif self.config.model_type == 'sa':
            from .models.sa_models import Generator
            self.generator = Generator(
                out_dim=self.config.input_shape[0],
                shapes=self.config.model_shapes,
                z_shape=latent_shape,
                filters=self.config.generator_filters,
            )
        else:
            raise NotImplementedError(
                f"{self.config.model_type} model is not implemented.")

        if self.config.model_type == 'normal':
            from .models.models import Discriminator
            self.discriminator = Discriminator(
                in_ch=self.config.input_shape[0],
                shapes=self.config.model_shapes[::-1],
                filters=self.config.discriminator_filters,
                is_self_attention=self.config.use_self_attention_d,
                is_minibatch_std=self.config.use_minibatch_std,
                is_spectral_norm=self.config.use_spectral_norm,
                is_conditional=self.config.use_conditional,
            ).to(self.device)
        elif self.config.model_type == 'small':
            from .models.small_models import Discriminator
            self.discriminator = Discriminator(
                in_ch=self.config.input_shape[0],
                shapes=self.config.model_shapes[::-1],
                filters=self.config.discriminator_filters,
                use_bn=self.config.use_bn_d,
                use_self_attention=self.config.use_self_attention_d,
                use_minibatch_std=self.config.use_minibatch_std,
                use_recon_loss=self.config.use_recon_loss,
                use_conditional=self.config.use_conditional,
                use_sn=self.config.use_sn_d,
                use_pooling=self.config.use_pooling_d
            ).to(self.device)
        elif self.config.model_type == 'sa':
            from .models.sa_models import Discriminator
            self.discriminator = Discriminator(
                in_ch=self.config.input_shape[0],
                shapes=self.config.model_shapes[::-1],
                filters=self.config.discriminator_filters,
            )
        else:
            raise NotImplementedError(
                f"{self.config.model_type} model is not implemented.")

        # Optimizer
        self.optimizer_g = torch.optim.RMSprop(
            self.generator.parameters(), lr=self.config.generator_lr
        )
        self.optimizer_d = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self.config.discriminator_lr
        )

        self.max_playable_count = 0
        self.reset_steps = 0
        self.bootstrap_count = 0

    def train(self):
        wandb.login()
        metrics = {}
        with wandb.init(project=f"{self.config.env_name} Level GAN by {self.config.model_type} model", config=self.config.__dict__):
            # check model summary
            self.generator.summary(batch_size=self.config.train_batch_size)
            self.discriminator.summary(
                batch_size=self.config.train_batch_size)

            # summaryによってdeviceが飛んでいるので注意
            self.generator.to(self.device)
            self.discriminator.to(self.device)

            step = 0
            self.latents_for_show = torch.randn(
                9, self.config.latent_size,).to(self.device)
            self.labels_for_show = self._get_labels(9).int()

            # model_save_path
            model_save_path = os.path.join(
                self.config.checkpoints_path,
                self.config.name + "-" + wandb.run.name.split("-")[-1],
            )
            os.makedirs(model_save_path)

            for epoch in range(1, self.config.epochs + 1):
                self.generator.train()
                for latent, real, label in self.train_loader:
                    step += 1
                    latent, real, label = (
                        latent.to(self.device).float(),
                        real.to(self.device).float(),
                        label.to(self.device).int(),
                    )

                    fake_logit = self.generator(latent, label)
                    Dx, DGz_d, discriminator_loss, recon_loss = self._discriminator_update(
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
                    if self.config.use_recon_loss:
                        metrics["Reconstruction Loss"] = recon_loss
                    metrics["Epoch"] = epoch
                    wandb.log(metrics)

                    if step >= self.config.steps:
                        break

                if step >= self.config.steps:
                    break

                self.generator.eval()
                with torch.no_grad():
                    if epoch % self.image_save_epoch == 0:
                        self._save_images()
                        if self.config.use_recon_loss:
                            self._save_recon_images(real, label)

                    if epoch % self.eval_epoch == 0 or epoch % self.bootstrap_epoch == 0:
                        # bootstrap and data-extend
                        latents = torch.randn(
                            self.config.eval_playable_counts,
                            self.config.latent_size,
                        ).to(self.device)
                        labels = self._get_labels(
                            self.config.eval_playable_counts).int()
                        levels = self._generate_levels(latents, labels)
                        playable_levels = []
                        for level_str in levels:
                            if check_playable(level_str):
                                playable_levels.append(level_str)
                        wandb.log(
                            {
                                "Playable Ratio": len(playable_levels)
                                / self.config.eval_playable_counts
                            }
                        )

                        # eval
                        if epoch % self.eval_epoch == 0:
                            self._eval_models(levels, playable_levels)

                        # bootstrap
                        unique_playable_levels = list(set(playable_levels))
                        if self.config.bootstrap and len(unique_playable_levels) > 1:
                            unique_playable_levels = self._bootstrap(
                                unique_playable_levels)
                        # if isinstance(self.config, DataExtendConfig):
                        #     for playable_level in unique_playable_levels:
                        #         with open(
                        #             os.path.join(self.config.level_data_path,
                        #                          self.config.env_name, "generated", f"{self.config.env_name}_{self.data_index}"), mode="w"
                        #         ) as f:
                        #             f.write(playable_level)
                        #             self.data_index += 1

                        # if isinstance(self.config, DataExtendConfig) and self.config.reset_weight_threshold < len(playable_levels)/self.config.eval_playable_counts:
                        if isinstance(self.config, DataExtendConfig) and self.bootstrap_count >= self.config.reset_weight_bootstrap_count:
                            print("reset weights of model.")
                            self._save_images()
                            if self.config.use_recon_loss:
                                self._save_recon_images(real, label)
                            self._build_model()
                            self.reset_steps = step

                    if isinstance(self.config, DataExtendConfig) and step-self.reset_steps >= 5000:
                        print(
                            f"reset weights of model. {step-self.reset_steps} steps progressed.")
                        self._build_model()
                        self.reset_steps = step

                    if epoch % self.model_save_epoch == 0:
                        self._save_models(
                            model_save_path, epoch, None)

    def _generate_levels(self, latents, labels):
        p_level = torch.softmax(
            self.generator(latents, labels), dim=1)
        level_strs = tensor_to_level_str(
            self.config.env_name, p_level)
        return level_strs

    def _save_images(self):
        p_level = torch.nn.Softmax2d()(
            self.generator(self.latents_for_show, self.labels_for_show))
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
        grid_level_img = make_grid(
            p_level_img, nrow=3, padding=0)
        images = wandb.Image(
            grid_level_img, caption="generated levels")
        wandb.log({"Generated Levels": images})

    def _save_recon_images(self, real, label):
        _, recon = self.discriminator(real[:9], label[:9])
        real_level_strs = tensor_to_level_str(
            self.config.env_name, real[:9]
        )
        real_level_img = [
            torch.Tensor(
                np.array(self.level_visualizer.draw_level(lvl)).transpose(
                    2, 0, 1
                )
                / 255.0
            )
            for lvl in real_level_strs
        ]
        recon_level_strs = tensor_to_level_str(
            self.config.env_name, recon
        )
        recon_level_img = [
            torch.Tensor(
                np.array(self.level_visualizer.draw_level(lvl)).transpose(
                    2, 0, 1
                )
                / 255.0
            )
            for lvl in recon_level_strs
        ]
        real_level_img = make_grid(
            real_level_img, nrow=3, padding=0)
        recon_level_img = make_grid(
            recon_level_img, nrow=3, padding=0)
        images = wandb.Image(
            real_level_img, caption="real levels")
        wandb.log({"Real Levels": images})
        images = wandb.Image(
            recon_level_img, caption="reconstructed levels")
        wandb.log({"Reconstructed Levels": images})

    def _eval_models(self, level_strs, playable_levels):

        (
            level_similarity,
            object_similarity,
            shape_similarity,
            duplicate_ratio,
        ) = self._calc_level_similarity(level_strs)

        wandb.log({"Level Similarity Ratio": level_similarity})
        wandb.log({"Special Object Similarity Ratio": object_similarity})
        wandb.log({"Shape Similarity Ratio": shape_similarity})
        wandb.log({"Duplicate Ratio": duplicate_ratio})

        if len(playable_levels) > self.max_playable_count:
            self.max_playable_count = len(playable_levels)
            self.playability = self.max_playable_count / self.config.eval_playable_counts

    # 生成データによる学習データ拡張

    def _bootstrap(self, unique_playable_levels: list[str]):
        if self.config.bootstrap == "random":
            list = random.sample(
                unique_playable_levels, min(len(unique_playable_levels), self.config.bootstrap_max_count))
            for level in list:
                self._level_expand(level)
        elif self.config.bootstrap == "smart":
            dataset_levels = []
            for file in self.dataset_files:
                with open(os.path.join(self.config.level_data_path, self.config.env_name, self.config.dataset_type, file), mode='r') as f:
                    s = f.read()
                    dataset_levels.append(s)
            result_levels = []
            if self.config.bootstrap_filter == None:
                result_levels = unique_playable_levels
            else:
                for generated_level in unique_playable_levels:
                    dup = False
                    for ds_level in dataset_levels:
                        sim = check_level_similarity(generated_level, ds_level)
                        if sim >= self.config.bootstrap_filter:
                            dup = True
                            break
                    if not dup:
                        result_levels.append(generated_level)

            selected = random.sample(
                result_levels, min(len(result_levels), self.config.bootstrap_max_count))
            for level in selected:
                self._level_add(level)
                self.bootstrap_count += 1

            wandb.log({"Dataset Size": len(self.dataset_files)})
            # update dataloader
            # Dataset
            del self.train_loader
            train_dataset = LevelDataset(
                self.config.level_data_path,
                self.env,
                datamode=self.config.dataset_type,
                transform=None,
                latent_size=self.config.latent_size,
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.config.train_batch_size, shuffle=True, drop_last=True
            )
            # self.steps_per_epoch = len(self.train_loader)
            # self.model_save_epoch = max(1, self.config.save_model_interval//(
            #     self.config.train_batch_size*self.steps_per_epoch))
            # self.image_save_epoch = max(1, self.config.save_image_interval//(
            #     self.config.train_batch_size*self.steps_per_epoch))
            # self.eval_epoch = max(1, self.config.eval_playable_interval//(
            #     self.config.train_batch_size*self.steps_per_epoch))
            # dataset files
            # self.dataset_files = os.listdir(train_dataset.image_dir)
            # print("Training Dataset :", train_dataset.image_dir)
            # DataLoader

            return result_levels
            # for i in range(len(playable_levels)):
            #     key = 0
            #     for j in range(len(playable_levels)):
            #         if i == j:
            #             continue
            #         key += check_level_similarity(
            #             playable_levels[i], playable_levels[j])
            #     playable_levels_pairs.append([playable_levels[i], key])

            # playable_levels_pairs.sort(key=operator.itemgetter(1))

            # exchange_count = min(
            #     self.config.bootstrap_max_count, len(playable_levels_pairs))

            # for i in range(exchange_count):
            #     index = np.random.randint(0, len(self.dataset_files)-5)
            #     file = self.dataset_files[index]
            #     with open(
            #         os.path.join(self.config.level_data_path,
            #                      self.config.env_name, self.config.dataset_type, file), mode="w"
            #     ) as f:
            #         f.write(playable_levels_pairs[i][0])
        else:
            NotImplementedError(
                f'{self.config.bootstrap} bootstrap is not implemented!!')

    def _level_add(self, level_str):
        index = len(self.dataset_files)
        file = f'{self.config.env_name}_{index}'
        self.dataset_files.append(file)
        with open(
            os.path.join(self.config.level_data_path,
                         self.config.env_name, self.config.dataset_type, file),
            "w",
        ) as f:
            f.write(level_str)

    def _discriminator_update(self, real_images_logit, fake_images_logit, label=None):
        """
        update discriminator: maximize log(D(x)) + log(1-D(G(z)))
        """
        self.discriminator.zero_grad()
        if self.config.use_recon_loss:
            real_logits, real_recon = self.discriminator(
                real_images_logit, label)
            fake_logits, _ = self.discriminator(
                torch.softmax(fake_images_logit, dim=1).detach(), label)
            real_logits, fake_logits = real_logits.view(
                -1), fake_logits.view(-1)
        else:
            real_logits = self.discriminator(
                real_images_logit, label).view(-1)
            fake_logits = self.discriminator(
                torch.softmax(fake_images_logit, dim=1).detach(), label).view(-1)

        if self.config.adv_loss == "baseline":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss(
                real_logits, fake_logits, self.config.smooth_label_value)
        elif self.config.adv_loss == "hinge":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss_hinge(
                real_logits, fake_logits
            )
        else:
            raise NotImplementedError()
        discriminator_loss = loss_real + loss_fake

        recon_loss_item = None
        if self.config.use_recon_loss:
            recon_loss = loss.recon_loss(real_recon, real_images_logit)
            discriminator_loss += self.config.recon_lambda*recon_loss
            recon_loss_item = recon_loss.item()
        discriminator_loss.backward()
        self.optimizer_d.step()

        return D_x, D_G_z, discriminator_loss.item(), recon_loss_item

    def _generator_update(self, fake_images_logit, label=None):
        """
        update generator: maximize log(D(G(z)))
        """
        self.generator.zero_grad()
        if self.config.use_recon_loss:
            fake_logits = self.discriminator(
                torch.softmax(fake_images_logit, dim=1), label)[0].view(-1)
        else:
            fake_logits = self.discriminator(
                torch.softmax(fake_images_logit, dim=1), label).view(-1)
        if self.config.adv_loss == "baseline":
            generator_loss = loss.g_loss(fake_logits)
        elif self.config.adv_loss == "hinge":
            generator_loss = loss.g_loss_hinge(fake_logits)

        div_loss = loss.div_loss(
            torch.softmax(fake_images_logit, dim=1), self.config.div_loss, self.config.lambda_div)
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
            os.path.join(model_save_path, f"models_{epoch}.tar"),
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
                         self.config.env_name, self.config.dataset_type, file),
            "w",
        ) as f:
            f.write(level_str)

    def _get_labels(self, n_labels):
        labels = []
        for i in range(n_labels):
            file = self.dataset_files[np.random.randint(
                len(self.dataset_files))]
            with open(os.path.join(self.config.level_data_path, self.config.env_name, self.config.dataset_type, file), "r") as f:
                datalist = f.readlines()
            # label : onehot vector of counts of map tile object.
            label = np.zeros(len(self.env.ascii))
            for i, s in enumerate(datalist):
                for j, c in enumerate(s):
                    if c == "\n":
                        break
                    label[self.env.ascii.index(c)] += 1
            labels.append(label)
        return torch.tensor(np.array(labels)).float().to(self.device)
