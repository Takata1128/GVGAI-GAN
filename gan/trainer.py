from __future__ import annotations
import wandb
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import os
import torch
import numpy as np
import random
import shutil
import statistics


from . import loss
from .game.env import Game
from .level_visualizer import GVGAILevelVisualizer, MarioLevelVisualizer
from .dataset import LevelDataset
from .level_dataset_extend import prepare_dataset
from .config import DataExtendConfig, BaseConfig
from .utils import (
    kmeans_select
)


class Trainer:
    def __init__(self, game: Game, models: dict, config: BaseConfig):
        self.config = config
        self.game = game

        # reproducible settings
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # device
        if config.cuda:
            self.device = torch.device(
                f"cuda:{config.gpu_id}" if torch.cuda.is_available else "cpu")
            print(f"device : cuda:{config.gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("device : cpu")

        # Dataset
        prepare_dataset(
            game, seed=config.seed, extend_data=config.clone_data, flip=config.flip_data, dataset_size=config.dataset_size)
        self.dataset = LevelDataset(
            config.level_data_path,
            self.game,
            latent_size=config.latent_size,
            use_diversity_sampling=config.use_diversity_sampling,
            seed=config.seed
        )

        similarities = []
        for i in range(len(self.dataset.data)):
            for j in range(i + 1, len(self.dataset.data)):
                level1 = self.dataset.data[i].representation
                level2 = self.dataset.data[j].representation
                similarities.append(self.game.check_similarity(level1, level2))

        similarities_mean = statistics.mean(similarities)
        similarities_stdev = statistics.stdev(similarities)

        self.hamming_filter_threshold = config.bootstrap_hamming_filter if self.config.bootstrap_hamming_filter else similarities_mean + 3 * similarities_stdev
        print("Hamming Filter :", self.hamming_filter_threshold)

        # Dataset files
        self.dataset_files = os.listdir(self.dataset.level_dir)
        print("Training Dataset :", self.dataset.level_dir)
        self.dataset_properties_dicts = self._init_dataset_properties()

        if isinstance(self.config, DataExtendConfig):
            generated_levels_path = self.config.generated_data_path
            if os.path.exists(generated_levels_path):
                shutil.rmtree(generated_levels_path)
            os.makedirs(generated_levels_path)

        # Level Visualizer
        if self.game.name == 'mario':
            self.level_visualizer = MarioLevelVisualizer(
                self.game, sprites_dir=self.config.data_path)
        else:
            self.level_visualizer = GVGAILevelVisualizer(self.game)

        # Network
        self.generator = models['generator']
        self.discriminator = models['discriminator']
        # Optimizer
        # self.optimizer_g = torch.optim.Adam(self.generator.parameters(),
        #                                     lr=self.config.generator_lr, betas=(0, 0.9))
        # self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(),
        #                                     lr=self.config.discriminator_lr, betas=(0, 0.9))
        self.optimizer_g = torch.optim.RMSprop(
            self.generator.parameters(), lr=self.config.generator_lr
        )
        self.optimizer_d = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self.config.discriminator_lr
        )

        # Other
        self.playability = 0
        self.augmented_data_index = 0
        self.max_playable_count = 0
        self.bootstrap_count = 0

        self.model_save_epoch = config.save_model_epoch
        self.image_save_epoch = config.save_image_epoch
        self.eval_epoch = config.eval_epoch
        self.bootstrap_epoch = config.bootstrap_epoch
        print(f"model_save_epoch:{self.model_save_epoch}")
        print(f"image_save_epoch:{self.image_save_epoch}")
        print(f"eval_epoch:{self.eval_epoch}")
        print(f"bootstrap_epoch:{self.bootstrap_epoch}")
        if isinstance(self.config, DataExtendConfig):
            self.reset_epoch = self.config.reset_weight_epoch
            print(f"model_reset_epoch:{self.reset_epoch}")
            self.last_reset_steps = 0
            self.training_set_reset_count = 0

    def train(self):
        wandb.login()
        metrics = {}
        with wandb.init(project=f"Generation of {self.config.env_fullname} Level by GAN", config=self.config.__dict__):
            # check model summary
            self.generator.summary(
                batch_size=self.config.train_batch_size, device=self.device)
            self.discriminator.summary(
                batch_size=self.config.train_batch_size, device=self.device)

            # model_save_path
            self.model_save_path = os.path.join(
                self.config.checkpoints_path,
                self.config.name + "-" + wandb.run.name.split("-")[-1],
            )

            os.makedirs(self.model_save_path)
            self.latents_for_show, _, self.labels_for_show = self.dataset.sample(
                9)
            self.latents_for_show = self.latents_for_show.to(
                self.device).float()
            self.labels_for_show = self.labels_for_show.to(self.device).int()

            self.step = 0

            while self.step < self.config.steps + 1:
                self.step += 1
                self.generator.train()

                ### Discriminator Update ###
                for _ in range(self.config.discriminator_update_count):
                    latent, real, label = self.dataset.sample(
                        batch_size=self.config.train_batch_size)
                    latent, real, label = (
                        latent.to(self.device).float(),
                        real.to(self.device).float(),
                        label.to(self.device).int(),
                    )
                    fake_images, hiddens = self.generator(latent, label)
                    Dx, DGz_d, discriminator_loss, recon_loss, gradient_penalty = self._discriminator_update(
                        real, fake_images, label
                    )

                ### Generator Update ###
                generator_loss, div_loss = self._generator_update(
                    latent, fake_images, label, hiddens)

                metrics = {}
                metrics["D(x)[Discriminator]"] = Dx
                metrics["D(G(z))[Discriminator]"] = DGz_d
                metrics["Discriminator Loss"] = discriminator_loss
                metrics["Generator Loss"] = generator_loss
                if self.config.div_loss:
                    metrics["Generator Div Loss"] = div_loss
                if self.config.use_recon_loss:
                    metrics["Reconstruction Loss"] = recon_loss
                if self.config.use_gradient_penalty:
                    metrics["Gradient Penalty"] = gradient_penalty
                wandb.log(metrics, step=self.step)
                if not self.after_step(real, label):
                    break
            self._evaluation()
            self._save_models(self.model_save_path, None, None)

    def after_step(self, real_images: torch.Tensor, labels: torch.Tensor):
        self.generator.eval()
        with torch.no_grad():
            if self.step % self.image_save_epoch == 0:
                self._save_images()
                if self.config.use_recon_loss:
                    self._save_recon_images(real_images, labels)

            if self.step % self.eval_epoch == 0 or (self.config.bootstrap and self.step % self.bootstrap_epoch == 0):
                # generate levels
                latents, _, labels = self.dataset.sample(
                    self.config.eval_playable_counts)
                latents = latents.to(self.device).float()
                labels = labels.to(self.device).int()
                levels = self._generate_levels(latents, labels)
                playable_levels = []
                for level_str in levels:
                    if self.game.check_playable(level_str):
                        playable_levels.append(level_str)
                wandb.log(
                    {
                        "Playable Rate": len(playable_levels)
                        / self.config.eval_playable_counts
                    }, step=self.step
                )

                # eval
                if self.step % self.eval_epoch == 0:
                    self._eval_models(levels, playable_levels)

                # save model
                if self.step % self.model_save_epoch == 0:
                    self._save_models(
                        self.model_save_path, self.step, None)

                # bootstrap
                unique_playable_levels = list(set(playable_levels))
                if self.step % self.bootstrap_epoch == 0 and self.config.bootstrap is not None and len(unique_playable_levels) > 1:
                    unique_playable_levels = self._bootstrap(
                        unique_playable_levels)

                # initialize process in augmentation model
                if isinstance(self.config, DataExtendConfig):
                    # 一定数生成したらGANの重みリセット
                    if self.bootstrap_count >= self.config.reset_weight_bootstrap_count:
                        print("reset weights of model.")
                        self._save_images()
                        # if self.config.use_recon_loss:
                        #     self._save_recon_images(real, label)
                        self._build_model()
                        self.last_reset_steps = self.step
                        self.bootstrap_count = 0
                    # 一定数生成したら学習データリセット
                    if self.augmented_data_index > self.config.reset_train_dataset_th * (self.training_set_reset_count + 1):
                        print(
                            f"Generated size exceeds {self.augmented_data_index}, so reset training dataset.")
                        self.training_set_reset_count += 1
                        prepare_dataset(
                            self.game, seed=self.config.seed, extend_data=self.config.clone_data, flip=self.config.flip_data
                        )
                        # initialize dataset
                        self.dataset.initialize()

                    # 一定期間たったらモデルを強制リセット
                    if self.step - self.last_reset_steps >= self.config.reset_weight_epoch:
                        print(
                            f"reset weights of model. {self.step-self.last_reset_steps} steps progressed.")
                        self._build_model()
                        self.last_reset_steps = self.step

                    # 全体である程度生成したらデータ拡張ストップ
                    if self.augmented_data_index >= self.config.stop_generate_count:
                        print(
                            f"Generated size exceeds {self.augmented_data_index}, so stop generation.")
                        return False
        return True

    def _generate_levels(self, latents: torch.Tensor, labels: torch.Tensor):
        level_tensor, _ = self.generator(latents, labels)
        level_strs = self.game.level_tensor_to_strs(
            level_tensor)
        return level_strs

    def _save_images(self):
        level_tensor, _ = self.generator(
            self.latents_for_show, self.labels_for_show)
        level_strs = self.game.level_tensor_to_strs(
            level_tensor)
        # playables = [self.game.check_playable(level) for level in level_strs]
        # visited_maps = [self.game.check_reachable(
        #     level.split())[1] for level in level_strs]

        p_level_img = [
            torch.tensor(
                np.array(self.level_visualizer.draw_level(lvl)).transpose(
                    2, 0, 1
                )
                / 255.0
            )
            for lvl in level_strs
        ]
        # # 非プレイアブルなステージを強調
        # for i, p in enumerate(playables):
        #     if not p:
        #         p_level_img[i][0] += 0.2
        grid_level_torch_img = make_grid(
            p_level_img, nrow=3, padding=0)
        grid_level_pil_img = transforms.functional.to_pil_image(
            grid_level_torch_img)
        images = wandb.Image(
            grid_level_pil_img, caption="generated levels")
        wandb.log({"Generated Levels": images}, self.step)

    def _save_recon_images(self, real, label):
        _, recon = self.discriminator(real[:9], label[:9])
        real_level_strs = self.game.level_tensor_to_strs(
            real[:9]
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
        recon_level_strs = self.game.level_tensor_to_strs(
            recon
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
        real_level_img = transforms.functional.to_pil_image(make_grid(
            real_level_img, nrow=3, padding=0))
        recon_level_img = transforms.functional.to_pil_image(make_grid(
            recon_level_img, nrow=3, padding=0))
        images = wandb.Image(
            real_level_img, caption="real levels")
        wandb.log({"Real Levels": images}, self.step)
        images = wandb.Image(
            recon_level_img, caption="reconstructed levels")
        wandb.log({"Reconstructed Levels": images}, self.step)

    def _eval_models(self, level_strs: list[str], playable_levels: list[str]):
        (
            level_similarity,
            duplication_rate,
        ) = self._calc_level_similarity(level_strs)

        wandb.log({"Level Similarity Rate": level_similarity}, self.step)
        wandb.log({"Duplication Rate": duplication_rate}, self.step)

        if len(playable_levels) > self.max_playable_count:
            self.max_playable_count = len(playable_levels)
            self.playability = self.max_playable_count / self.config.eval_playable_counts

    # 生成データによる学習データ拡張

    def _bootstrap(self, unique_playable_levels: list[str]):
        if self.config.bootstrap == "random":
            result_levels = unique_playable_levels
        elif self.config.bootstrap == 'baseline':  # PCA + elbow + kmeansで追加するデータを決定
            result_levels = kmeans_select(
                unique_playable_levels, self.config, self.game)
        elif self.config.bootstrap == "smart":  # データセット内のステージと比較して類似度が低いもののみを追加
            # 類似度フィルタ
            # if self.config.bootstrap_hamming_filter == None:
            #     filtered_levels = unique_playable_levels
            # else:
            # データセットのレベルたち
            dataset_levels = []
            for item in self.dataset.data:
                dataset_levels.append(item.representation)
            filtered_levels = []
            for generated_level in unique_playable_levels:
                dup = False
                for ds_level in dataset_levels:
                    sim = self.game.check_similarity(
                        generated_level, ds_level)
                    # if sim >= self.config.bootstrap_hamming_filter:
                    if sim >= self.hamming_filter_threshold:
                        dup = True
                        break
                if not dup:
                    filtered_levels.append(generated_level)

            print(f"Filtered by Hamming: {len(filtered_levels)}")

            # プロパティフィルタ
            if self.config.bootstrap_property_filter is not None:
                tmp_levels = []
                for level_str in filtered_levels:
                    features = self.game.get_features(level_str)
                    ok = True
                    for i, pd in enumerate(self.dataset_properties_dicts):
                        if (features[i] in pd) and (pd[features[i]] >= self.dataset.data_length * self.config.bootstrap_property_filter):
                            print(
                                f"{pd[features[i]]} >= {self.dataset.data_length * self.config.bootstrap_property_filter}")
                            ok = False
                    if ok:
                        tmp_levels.append(level_str)

                filtered_levels = tmp_levels
            else:
                pass

            print("Property : ")
            for feature, v in self.dataset.feature2indices.items():
                print(f"{feature}={len(v)} , ", end="")
            print()
            print(f"Filtered by features: {len(filtered_levels)}")
            print()

            # K-meansで選択
            if self.config.bootstrap_kmeans_filter and len(filtered_levels) > 1:
                result_levels = kmeans_select(
                    filtered_levels, self.config, self.game)
            else:
                result_levels = filtered_levels
        else:
            raise NotImplementedError(
                f'{self.config.bootstrap} bootstrap is not implemented!!')
        # 多すぎないように
        selected = random.sample(
            result_levels, min(len(result_levels), self.config.bootstrap_max_count))
        for level in selected:
            self._level_add(level)
            self.bootstrap_count += 1
            # generatedフォルダにも追加
            if isinstance(self.config, DataExtendConfig):
                with open(
                    os.path.join(self.config.generated_data_path, f"{self.config.env_name}_{self.augmented_data_index}"), mode="w"
                ) as f:
                    f.write(level)
                    self.augmented_data_index += 1
        wandb.log({"Dataset Size": len(self.dataset_files)}, self.step)
        # Update dataset
        return selected

    def _level_add(self, level_str: str):
        index = len(self.dataset_files)
        file = f'{self.config.env_name}_{index}'
        self.dataset_files.append(file)
        with open(
            os.path.join(self.config.level_data_path, file),
            "w",
        ) as f:
            f.write(level_str)
            feature = self.game.get_features(level_str)
            self.dataset.add_data(level_str, feature)
            for i, pd in enumerate(self.dataset_properties_dicts):
                if feature[i] in pd:
                    pd[feature[i]] += 1
                else:
                    pd[feature[i]] = 1

    def _discriminator_update(self, real_images: torch.Tensor, fake_images: torch.Tensor, label: torch.Tensor = None):
        """
        update discriminator: maximize log(D(x)) + log(1-D(G(z)))
        """
        def mix(real, fake):
            for i in range(real.size(0)):
                if np.random.random() < 0.05:
                    real[i, :] = fake[i, :]
            return real.detach()

        self.discriminator.zero_grad()
        if self.config.use_recon_loss:
            real_logits, real_recon = self.discriminator(
                real_images, label)
            fake_logits, _ = self.discriminator(
                fake_images.detach(), label)
            real_logits, fake_logits = real_logits.view(
                -1), fake_logits.view(-1)
        else:
            real_logits = self.discriminator(
                real_images, label).view(-1)
            fake_logits = self.discriminator(
                fake_images.detach(), label).view(-1)

        if self.config.adv_loss == "baseline":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss(
                real_logits, fake_logits)
        elif self.config.adv_loss == "hinge":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss_hinge(
                real_logits, fake_logits
            )
        elif self.config.adv_loss == "wgan":
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss_wgan(
                real_logits, fake_logits
            )
        elif self.config.adv_loss == 'wgan-gp':
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss_wgan(
                real_logits, fake_logits
            )
        elif self.config.adv_loss == 'lsgan':
            loss_real, loss_fake, D_x, D_G_z = loss.d_loss_lsgan(
                real_logits, fake_logits
            )
        else:
            raise NotImplementedError()

        discriminator_loss = loss_real + loss_fake

        gradient_penalty_item = None
        if self.config.use_gradient_penalty:
            # gradient_penalty
            gradient_penalty = loss.calc_gradient_penalty(
                real_images, fake_images, self.discriminator)
            discriminator_loss += self.config.gp_lambda * gradient_penalty
            gradient_penalty_item = gradient_penalty.item()

        recon_loss_item = None
        if self.config.use_recon_loss:
            recon_loss = loss.recon_loss(real_recon, real_images)
            discriminator_loss += self.config.recon_lambda * recon_loss
            recon_loss_item = recon_loss.item()

        discriminator_loss.backward(retain_graph=True)
        self.optimizer_d.step()

        if self.config.use_clipping_d:
            # clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

        return D_x, D_G_z, discriminator_loss.item(), recon_loss_item, gradient_penalty_item

    def _generator_update(self, latent: torch.Tensor, fake_images: torch.Tensor, label: torch.Tensor = None, hiddens: torch.Tensor = None):
        """
        update generator: maximize log(D(G(z)))
        """
        self.generator.zero_grad()
        if self.config.use_recon_loss:
            fake_logits = self.discriminator(
                fake_images, label)[0].view(-1)
        else:
            fake_logits = self.discriminator(
                fake_images, label).view(-1)
        if self.config.adv_loss == "baseline":
            generator_loss = loss.g_loss(fake_logits)
        elif self.config.adv_loss == "hinge":
            generator_loss = loss.g_loss_hinge(fake_logits)
        elif self.config.adv_loss == 'wgan':
            generator_loss = loss.g_loss_wgan(fake_logits)
        elif self.config.adv_loss == 'wgan-gp':
            generator_loss = loss.g_loss_wgan(fake_logits)
        elif self.config.adv_loss == 'lsgan':
            generator_loss = loss.g_loss_lsgan(fake_logits)
        else:
            raise NotImplementedError()
        div_loss = loss.div_loss(
            latent, fake_images, hiddens, self.config.div_loss, self.config.lambda_div, self.game)
        generator_loss += div_loss
        generator_loss.backward(retain_graph=True)
        self.optimizer_g.step()

        return generator_loss.item(), div_loss.item()

    def _save_models(self, model_save_dir: str, epoch: int, playable_count: int):
        model_save_path = os.path.join(
            model_save_dir, f"models_{epoch}.tar" if epoch is not None else "latest.tar")
        torch.save(
            {
                "epoch": epoch,
                "playable_count": playable_count,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            model_save_path
        )

    def _calc_level_similarity(self, level_strs: list[str]):
        res_level, res_duplicate, n_level = 0, 0, 0
        for i in range(0, len(level_strs)):
            for j in range(i + 1, len(level_strs)):
                sim = self.game.check_similarity(
                    level_strs[i], level_strs[j])
                res_level += sim
                res_duplicate += (
                    1
                    if sim >= 0.90
                    else 0
                )
                n_level += 1
        return (
            res_level / n_level,
            res_duplicate / n_level,
        )

    def _evaluation(self):
        playable_levels = []
        for i in range(100):
            latents_for_eval, _, labels_for_eval = self.dataset.sample(5000)
            latents_for_eval, labels_for_eval = latents_for_eval.to(
                self.device).float(), labels_for_eval.to(self.device).int()
            output_levels, _ = self.generator(
                latents_for_eval, labels_for_eval)
            level_strs = self.game.level_tensor_to_strs(output_levels)
            for level_str in level_strs:
                if self.game.check_playable(level_str):
                    playable_levels.append(level_str)
                    if len(playable_levels) == self.config.final_evaluation_levels:
                        break
            print(len(playable_levels))
            if len(playable_levels) == self.config.final_evaluation_levels:
                break

        if len(playable_levels) == self.config.final_evaluation_levels:
            metrics = self.game.evaluation(playable_levels)
        else:
            metrics = {}

        playable_levels = []
        latents_for_eval, _, labels_for_eval = self.dataset.sample(
            self.config.final_evaluation_levels)
        latents_for_eval, labels_for_eval = latents_for_eval.to(
            self.device).float(), labels_for_eval.to(self.device).int()
        output_levels, _ = self.generator(latents_for_eval, labels_for_eval)
        level_strs = self.game.level_tensor_to_strs(output_levels)
        for level_str in level_strs:
            if self.game.check_playable(level_str):
                playable_levels.append(level_str)

        metrics["Final Playable Rate"] = len(
            playable_levels) / self.config.final_evaluation_levels
        print("Playable Rate:", len(playable_levels) /
              self.config.final_evaluation_levels)
        wandb.log(metrics)

    def _init_dataset_properties(self):
        ret = []
        image_paths = [
            os.path.join(self.dataset.level_dir, name) for name in os.listdir(self.dataset.level_dir)
        ]
        for path in image_paths:
            level_str = ''
            with open(path, mode='r') as f:
                level_str = f.read()
            property = self.game.get_features(level_str)
            # ret[property] += 1
            for i, p in enumerate(property):
                if i >= len(ret):
                    ret.append({})
                if p in ret[i]:
                    ret[i][p] += 1
                else:
                    ret[i][p] = 1
        return ret
