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
    def __init__(self, dataset_dir: str, game: Game, latent_size: int = 32, diversity_sampling_mode: str = 'legacy', initial_data_featuring=False, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.game = game
        self.level_dir = dataset_dir
        self.latent_size = latent_size
        self.diversity_sampling_mode = diversity_sampling_mode
        self.initial_data_featuring = initial_data_featuring
        self.initialize()

    def initialize(self):
        self.data: list[LevelItem] = []
        self.feature2indices = {}
        self.data_index = 0
        self.level_paths = [os.path.join(
            self.level_dir, name) for name in os.listdir(self.level_dir)]
        self.data_length = len(self.level_paths)
        self.feature_search_dicts = []
        for path in self.level_paths:
            with open(path, 'r') as f:
                level = f.read()
            features = self.game.get_features(level)

            # 初期ステージを重視（特徴量にindexを加える）
            if self.initial_data_featuring:
                features = features + (len(self.data) + 1,)
            level_tensor, label_tensor = self.game.level_str_to_tensor(level)
            item = LevelItem(level_tensor, label_tensor, level, features)

            # 特徴量とステージをマッピング
            if features in self.feature2indices:
                self.feature2indices[features].append(len(self.data))
            else:
                self.feature2indices[features] = [len(self.data)]

            ### proposal 特徴量を単純にランダムに選ぶのではなく、あるひとつに着目してランダムに選ぶ ###
            if self.feature_search_dicts == []:
                for _ in range(len(features)):
                    self.feature_search_dicts.append(dict())
            for i in range(len(features)):
                if features[i] in self.feature_search_dicts[i]:
                    self.feature_search_dicts[i][features[i]].add(features)
                else:
                    self.feature_search_dicts[i][features[i]] = {features}
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
        for i in range(len(features)):
            if features[i] in self.feature_search_dicts[i]:
                self.feature_search_dicts[i][features[i]].add(features)
            else:
                self.feature_search_dicts[i][features[i]] = {features}
        self.data.append(item)
        self.data_length = len(self.data)

    def diversity_sampling(self, step):
        feature_index = np.random.choice(len(self.feature2indices))
        level_index = np.random.choice(self.feature2indices[list(
            self.feature2indices.keys())[feature_index]])
        item = self.data[level_index]
        return item

    def proposal_diversity_sampling(self, step):
        dict_index = np.random.choice(len(self.feature_search_dicts))
        features_set = random.choice(
            list(self.feature_search_dicts[dict_index].values()))
        feature = random.choice(list(features_set))
        level_index = np.random.choice(self.feature2indices[feature])
        item = self.data[level_index]
        return item

    def sample(self, batch_size: int, step=None):
        latent_batch = torch.zeros((batch_size, self.latent_size))
        level_batch = torch.zeros((batch_size, *self.game.input_shape))
        label_batch = torch.zeros((batch_size, self.game.input_shape[0]))
        # batch_features = {}
        for index in range(batch_size):
            if self.diversity_sampling_mode == 'legacy':
                item = self.diversity_sampling(step)
            elif self.diversity_sampling_mode == 'proposal':
                item = self.proposal_diversity_sampling(step)
            else:
                item = np.random.choice(self.data)
            latent_batch[index] = torch.randn(self.latent_size)
            level_batch[index] = item.data
            label_batch[index] = item.label
        return latent_batch, level_batch, label_batch
