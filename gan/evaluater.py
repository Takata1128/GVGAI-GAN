from argparse import ArgumentParser
import argparse
from pyexpat import model
from turtle import shape
from typing import List
from torchinfo import summary
from utils import (
    tensor_to_level_str,
    check_playable_zelda,
    check_level_similarity,
    check_object_similarity,
    check_shape_similarity,
)
import wandb
from dataset import LevelDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from level_visualizer import GVGAILevelVisualizer
import os
import torch
import numpy as np
import loss
from config import EvaluationConfig

from models import ShapeGenerator, Discriminator

# from new_models import Generator, Discriminator
from level_dataset_extend import prepare_dataset


class Evaluater:
    def __init__(self, config: EvaluationConfig, model_save_path: str):
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

        # Level Visualizer
        self.level_visualizer = GVGAILevelVisualizer(config.env_name)

        # Network
        latent_shape = (config.latent_size,)
        self.generator = ShapeGenerator(
            out_dim=config.input_shape[0],
            shapes=config.model_shapes,
            z_shape=latent_shape,
            filters=config.generator_filters,
            is_self_attention=config.is_self_attention_g,
            is_conditional=config.is_conditional,
        ).to(self.device)

    def evaluate_list(self, model_path_list: list):
        playable_ratios = []
        level_similarities = []
        duplicate_ratios = []

        for path in model_path_list:
            playable, level_sim, obj_sim, shape_sim, dup = self.evaluate(path)
            playable_ratios.append(playable)
            level_similarities.append(level_sim)
            duplicate_ratios.append(dup)
            # level_similarity += level_sim
            # object_similarity += obj_sim
            # shape_similarity += shape_sim
            # duplicate_ratio += dup

        playable_ratios = np.array(playable_ratios)
        level_similarities = np.array(level_similarities)
        duplicate_ratios = np.array(duplicate_ratios)

        print(
            "[Playable Ratio] {:.1f}/{:.1f}/{:.1f}".format(
                playable_ratios.min() * 100.0,
                playable_ratios.mean() * 100.0,
                playable_ratios.max() * 100.0,
            )
        )
        print(
            "[Level Similarity] {:.1f}/{:.1f}/{:.1f}".format(
                level_similarities.min() * 100.0,
                level_similarities.mean() * 100.0,
                level_similarities.max() * 100.0,
            )
        )
        print(
            "[Duplicate Ratio] {:.1f}/{:.1f}/{:.1f}".format(
                duplicate_ratios.min() * 100.0,
                duplicate_ratios.mean() * 100.0,
                duplicate_ratios.max() * 100.0,
            )
        )
        # print("[Level Similarity] :", level_similarities.mean)
        # print("Object Similarity :", object_similarity)
        # print("Shape Similarity :", shape_similarity)
        # print("[Duplicate Ratio] :", duplicate_ratios.mean)

    def evaluate(self, model_path: str):
        self._load_models(
            model_save_path=model_path,
        )

        latents_for_eval = torch.randn(
            self.config.eval_playable_counts,
            self.config.latent_size,
        ).to(self.device)

        labels_for_eval = []
        for i in range(self.config.eval_playable_counts // 5):
            labels_for_eval.append([100, 86, 1, 1, 1, 1, 1, 1])
            labels_for_eval.append([110, 76, 1, 1, 1, 1, 1, 1])
            labels_for_eval.append([120, 66, 1, 1, 1, 1, 1, 1])
            labels_for_eval.append([130, 56, 1, 1, 1, 1, 1, 1])
            labels_for_eval.append([140, 46, 1, 1, 1, 1, 1, 1])
        labels_for_eval = torch.tensor(
            np.array(labels_for_eval)).int().to(self.device)

        output_levels = self.generator(latents_for_eval, labels_for_eval)

        level_strs = tensor_to_level_str(self.config.env_name, output_levels)
        playable_count = 0
        for level_str in level_strs:
            if check_playable_zelda(level_str):
                playable_count += 1

        playable_ratio = playable_count / self.config.eval_playable_counts
        (
            level_similarity,
            object_similarity,
            shape_similarity,
            duplicate_ratio,
        ) = self._calc_level_similarity(level_strs)

        return (
            playable_ratio,
            level_similarity,
            object_similarity,
            shape_similarity,
            duplicate_ratio,
        )

    def _load_models(self, model_save_path: str):
        model_dict = torch.load(os.path.join(model_save_path, "models.tar"))
        self.generator.load_state_dict(model_dict["generator"])

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
            res_shape / n_shape_level if n_object_level > 0 else None,
            res_duplicate / n_level,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="none")
    args = parser.parse_args()
    config = EvaluationConfig()
    config.name = args.name
    evaluater = Evaluater(
        config, model_save_path="/root/mnt/ZeldaGan/scripts/gan/checkpoints"
    )
    print("L2-100.0")
    eval_paths = [
        "/root/mnt/ZeldaGan/scripts/gan/checkpoints/L2-div-100.0-514",
        "/root/mnt/ZeldaGan/scripts/gan/checkpoints/L2-div-100.0-515",
        "/root/mnt/ZeldaGan/scripts/gan/checkpoints/L2-div-100.0-516",
        "/root/mnt/ZeldaGan/scripts/gan/checkpoints/L2-div-100.0-517",
        "/root/mnt/ZeldaGan/scripts/gan/checkpoints/L2-div-100.0-518",
    ]
    evaluater.evaluate_list(model_path_list=eval_paths)
    evaluater = Evaluater(
        config, model_save_path="/root/mnt/ZeldaGan/scripts/gan/checkpoints"
    )
