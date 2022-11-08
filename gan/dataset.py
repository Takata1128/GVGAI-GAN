from dataclasses import dataclass
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch

import numpy as np

from .game.env import Game


@dataclass
class LevelItem:
    representation: str
    features: tuple


class LevelDataset(Dataset):
    def __init__(self, dataset_dir: str, game: Game, latent_size: int = 32):
        self.game = game
        self.level_dir = dataset_dir
        self.level_paths = [os.path.join(
            self.level_dir, name) for name in os.listdir(self.level_dir)]
        self.data_length = len(self.level_paths)
        self.latent_size = latent_size
        self.data: list[LevelItem] = []
        self.data_index = 0
        self.initialize()

    def initialize(self):
        for path in self.level_paths:
            with open(path, 'r') as f:
                level = f.read()
            features = self.game.get_property(level)
            self.data.append(LevelItem(level, features))

    def update(self):
        '''
        datasetの更新
        後に作ったレベルのファイル名が辞書順で後ろになることを仮定
        '''
        self.level_paths = [os.path.join(
            self.level_dir, name) for name in os.listdir(self.level_dir)]

        for index in range(self.data_length, len(self.level_paths)):
            path = self.level_paths[index]
            with open(path, 'r') as f:
                level = f.read()
            features = self.game.get_property(level)
            self.data.append(LevelItem(level, features))

        self.data_length = len(self.level_paths)

    def sample(self, batch_size: int):
        latent_batch = torch.zeros((batch_size, self.latent_size))
        level_batch = torch.zeros((batch_size, *self.game.input_shape))
        label_batch = torch.zeros((batch_size, self.game.input_shape[0]))
        for index in range(batch_size):
            item = np.random.choice(self.data)
            level_tensor, label_tensor = self.to_tensor(item.representation)
            latent_batch[index] = torch.randn(self.latent_size)
            level_batch[index] = level_tensor
            label_batch[index] = label_tensor
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


# class LevelDataset(Dataset):
#     def __init__(
#         self,
#         root,
#         env: Game,
#         datamode="train",
#         transform=transforms.ToTensor(),
#         latent_size=32,
#     ):
#         self.env = env
#         self.image_dir = os.path.join(
#             root, datamode)
#         self.image_paths = [
#             os.path.join(self.image_dir, name) for name in os.listdir(self.image_dir)
#         ]
#         self.data_length = len(self.image_paths)
#         self.transform = transform
#         self.latent_size = latent_size

#     def __len__(self):
#         return self.data_length

#     def __getitem__(self, index):
#         latent = torch.randn(size=(self.latent_size,))
#         img_path = self.image_paths[index]
#         img, label = self._open(img_path)

#         if not self.transform is None:
#             img = self.transform(img)

#         return latent, img, label

#     def _open(self, img_path):
#         with open(img_path, "r") as f:
#             level_str = f.readlines()
#         ret = np.zeros(
#             (len(self.env.ascii),
#              self.env.state_shape[1], self.env.state_shape[2]),
#         )

#         # padding
#         ret[1, :, :] = 1

#         # label : onehot vector of counts of map tile object.
#         label = np.zeros(len(self.env.ascii))
#         for i, s in enumerate(level_str):
#             for j, c in enumerate(s):
#                 if c == "\n":
#                     break
#                 ret[1, i, j] = 0
#                 ret[self.env.ascii.index(c), i, j] = 1
#                 label[self.env.ascii.index(c)] += 1
#         return ret, label
