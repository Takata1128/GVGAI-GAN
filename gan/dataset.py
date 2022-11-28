from dataclasses import dataclass
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch

import numpy as np

from .game.env import Game


@dataclass
class LevelItem:
    data: torch.Tensor
    label: torch.Tensor
    representation: str
    features: tuple


class LevelDataset(Dataset):
    def __init__(self, dataset_dir: str, game: Game, latent_size: int = 32, use_diversity_sampling=False):
        self.game = game
        self.level_dir = dataset_dir
        self.latent_size = latent_size
        self.use_diversity_sampling = use_diversity_sampling
        self.initialize()

    def initialize(self):
        self.data: list[LevelItem] = []
        self.feature2indices = {}
        self.data_index = 0
        self.level_paths = [os.path.join(
            self.level_dir, name) for name in os.listdir(self.level_dir)]
        self.data_length = len(self.level_paths)
        for path in self.level_paths:
            with open(path, 'r') as f:
                level = f.read()
            features = self.game.get_property(level)
            level_tensor, label_tensor = self.to_tensor(level)
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
        level_tensor, label_tensor = self.to_tensor(level)
        item = LevelItem(level_tensor, label_tensor, level, features)
        if features in self.feature2indices:
            self.feature2indices[features].append(index)
        else:
            self.feature2indices[features] = [index]
        self.data.append(item)
        self.data_length = len(self.data)

    # def update(self):
    #     '''
    #     datasetの更新
    #     後に作ったレベルのファイル名が辞書順で後ろになることを仮定
    #     '''

    #     for index in range(self.data_length, len(self.level_paths)):
    #         path = self.level_paths[index]
    #         with open(path, 'r') as f:
    #             level = f.read()
    #         features = self.game.get_property(level)
    #         level_tensor, label_tensor = self.to_tensor(level)
    #         item = LevelItem(level_tensor, label_tensor, level, features)
    #         if features in self.feature2indices:
    #             self.feature2indices[features].append(index)
    #         else:
    #             self.feature2indices[features] = [index]
    #         self.data.append(item)

    #     self.data_length = len(self.data)

    def sample(self, batch_size: int):
        latent_batch = torch.zeros((batch_size, self.latent_size))
        level_batch = torch.zeros((batch_size, *self.game.input_shape))
        label_batch = torch.zeros((batch_size, self.game.input_shape[0]))
        batch_features = {}
        for index in range(batch_size):
            if self.use_diversity_sampling:
                key_index = np.random.choice(len(self.feature2indices))
                idx = np.random.choice(self.feature2indices[list(
                    self.feature2indices.keys())[key_index]])
                item = self.data[idx]
                if item.features in batch_features:
                    batch_features[item.features] += 1
                else:
                    batch_features[item.features] = 0
            else:
                item = np.random.choice(self.data)
            latent_batch[index] = torch.randn(self.latent_size)
            level_batch[index] = item.data
            label_batch[index] = item.label
        # for key, value in batch_features.items():
        #     print(f'{key} : {value}', end=', ')
        # print()
        return latent_batch, level_batch, label_batch

    def to_tensor(self, level: str):
        level = level.split()
        level_numpy = np.zeros(self.game.input_shape)
        # padding
        level_numpy[1, :, :] = 1
        # label : onehot vector of counts of map tile object.
        label_numpy = np.zeros(len(self.game.ascii))
        for i, s in enumerate(level):
            for j, c in enumerate(s):
                if c == "\n":
                    break
                level_numpy[1, i, j] = 0
                level_numpy[self.game.ascii.index(c), i, j] = 1
                label_numpy[self.game.ascii.index(c)] += 1
        return torch.tensor(level_numpy), torch.tensor(label_numpy)


class LevelDatasetOld(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        game: Game,
        latent_size: int,
    ):
        self.env = game
        self.image_dir = dataset_dir
        self.image_paths = [
            os.path.join(self.image_dir, name) for name in os.listdir(self.image_dir)
        ]
        self.data_length = len(self.image_paths)
        self.latent_size = latent_size

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        latent = torch.randn(size=(self.latent_size,))
        img_path = self.image_paths[index]
        img, label = self._open(img_path)
        return latent, img, label

    def _open(self, img_path):
        with open(img_path, "r") as f:
            level_str = f.readlines()
        ret = torch.zeros(
            (len(self.env.ascii),
             self.env.input_shape[1], self.env.input_shape[2]),
        )
        # padding
        ret[1, :, :] = 1
        # label : onehot vector of counts of map tile object.
        label = torch.zeros(len(self.env.ascii))
        for i, s in enumerate(level_str):
            for j, c in enumerate(s):
                if c == "\n":
                    break
                ret[1, i, j] = 0
                ret[self.env.ascii.index(c), i, j] = 1
                label[self.env.ascii.index(c)] += 1
        return ret, label
