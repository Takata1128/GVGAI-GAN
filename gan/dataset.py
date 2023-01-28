from dataclasses import dataclass
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch

import random
import numpy as np

from .game.env import Game


@dataclass
class LevelItem:
    data: torch.Tensor
    label: torch.Tensor
    representation: str
    features: tuple


class LevelDataset(Dataset):
    def __init__(self, dataset_dir: str, game: Game, latent_size: int = 32, use_diversity_sampling=False, initial_data_prob=0.0, initial_data_sampling_steps=None, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.game = game
        self.level_dir = dataset_dir
        self.latent_size = latent_size
        self.use_diversity_sampling = use_diversity_sampling
        self.initial_dataset_size = 0
        self.select_initial_data_prob = initial_data_prob
        self.data_noise_coef = 0.00
        self.initial_data_sampling_steps = initial_data_sampling_steps
        self.initialize()

    def initialize(self):
        self.data: list[LevelItem] = []
        self.feature2indices = {}
        self.data_index = 0
        self.level_paths = [os.path.join(
            self.level_dir, name) for name in os.listdir(self.level_dir)]
        self.data_length = len(self.level_paths)
        self.initial_dataset_size = self.data_length
        for path in self.level_paths:
            with open(path, 'r') as f:
                level = f.read()
            features = self.game.get_features(level)
            level_tensor, label_tensor = self.game.level_str_to_tensor(level)
            item = LevelItem(level_tensor, label_tensor, level, features)
            if features in self.feature2indices:
                self.feature2indices[features].append(len(self.data))
            else:
                self.feature2indices[features] = [len(self.data)]
            self.data.append(item)

        print('### Properties ###')
        for key, value in self.feature2indices.items():
            print(f'{key} : {len(value)}', end=', ')
        print()

    def add_data(self, level, features):
        index = self.data_length
        level_tensor, label_tensor = self.game.level_str_to_tensor(level)
        item = LevelItem(level_tensor, label_tensor, level, features)
        if features in self.feature2indices:
            self.feature2indices[features].append(index)
        else:
            self.feature2indices[features] = [index]
        self.data.append(item)
        self.data_length = len(self.data)

    def sample(self, batch_size: int, step=None):
        latent_batch = torch.zeros((batch_size, self.latent_size))
        level_batch = torch.zeros((batch_size, *self.game.input_shape))
        label_batch = torch.zeros((batch_size, self.game.input_shape[0]))
        # batch_features = {}
        for index in range(batch_size):
            if self.use_diversity_sampling:
                key_index = np.random.choice(len(self.feature2indices))
                idx = np.random.choice(self.feature2indices[list(
                    self.feature2indices.keys())[key_index]])
                # 初期データの割合が低くなり始めたら初期データ選択確率を保証
                if self.select_initial_data_prob and self.initial_dataset_size / self.data_length < self.select_initial_data_prob:
                    if step and self.initial_data_sampling_steps and step > self.initial_data_sampling_steps:
                        # 規定のステップ数を超えたらランダム選択
                        item = np.random.choice(self.data)
                    else:
                        if np.random.random() < self.select_initial_data_prob:
                            idx = np.random.randint(
                                self.initial_dataset_size)  # 初期データセットの中からランダム選択
                            item = self.data[idx]
                        else:
                            item = np.random.choice(self.data)
                else:
                    item = np.random.choice(self.data)
                item = self.data[idx]
                # if item.features in batch_features:
                #     batch_features[item.features] += 1
                # else:
                #     batch_features[item.features] = 0
            else:
                item = np.random.choice(self.data)

            latent_batch[index] = torch.randn(self.latent_size)
            level_batch[index] = item.data
            label_batch[index] = item.label
        # for key, value in batch_features.items():
        #     print(f'{key} : {value}', end=', ')
        # print()
        noise = torch.rand_like(level_batch)
        level_batch = level_batch + self.data_noise_coef * noise
        return latent_batch, level_batch, label_batch
