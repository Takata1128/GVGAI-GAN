from __future__ import annotations
from glob import glob
from abc import ABCMeta, abstractmethod
from unittest.mock import NonCallableMagicMock
import gym_gvgai
import torch
import numpy as np
import os

GameDescription = {}
GameDescription["aliens_v0"] = {
    "ascii": [".", "0", "1", "2", "A"],
    "state_shape": (5, 32, 32),
}
GameDescription["zelda_v1"] = {
    "ascii": [".", "w", "g", "+", "1", "2", "3", "A"],
    'char_to_tile': {
        '.': ['floor'],
        'w': ['wall'],
        'g': ['floor', 'goal'],
        '+': ['floor', 'key'],
        '1': ['floor', 'monsterQuick'],
        '2': ['floor', 'monsterNormal'],
        '3': ['floor', 'monsterSlow'],
        'A': ['floor', 'nokey'],
    },
    "state_shape": (8, 16, 16),
    "map_shape": (12, 16),
}
GameDescription["zelda_v0"] = {
    "ascii": [".", "w", "g", "+", "1", "2", "3", "A"],
    'char_to_tile': {
        '.': ['floor'],
        'w': ['wall'],
        'g': ['floor', 'goal'],
        '+': ['floor', 'key'],
        '1': ['floor', 'monsterQuick'],
        '2': ['floor', 'monsterNormal'],
        '3': ['floor', 'monsterSlow'],
        'A': ['floor', 'nokey'],
    },
    "state_shape": (8, 16, 16),
    "map_shape": (9, 13),
}

GameDescription["roguelike_v0"] = {
    "ascii": [".", 'w', "x", "s", "g", "r", "p", "h", "k", 'l', 'm', 'A'],
    'char_to_tile': {
        '.': ['floor'],
        'w': ['wall'],
        'x': ["floor", "exit"],
        's': ["floor", 'weapon'],
        'g': ["floor", 'gold'],
        'r': ["floor", "spider"],
        'p': ["floor", "phantom"],
        'h': ["floor", "health"],
        'k': ["floor", "key"],
        'l': ["floor", "lock"],
        'm': ["floor", "market"],
        'A': ["floor", "avatar"],
    },
    "state_shape": (12, 32, 32),
    "map_shape": (22, 23),
}

GameDescription["boulderdash_v0"] = {
    "ascii": ['w', ".", '-', "e", "o", "x", "c", "b", "A"],
    'char_to_tile': {
        'w': ['wall'],
        '.': ['background', 'dirt'],
        '-': ['background'],
        'e': ["background", "exitdoor"],
        'o': ["background", 'boulder'],
        'x': ["background", 'diamond'],
        'c': ["background", "crab"],
        'b': ["background", "butterfly"],
        'A': ["background", "avatar"],
    },
    "state_shape": (9, 32, 32),
    "map_shape": (13, 26),
}

GameDescription["mario_v0"] = {
    "ascii": ["X", 'S', '-', "Q", "E", "<", ">", "[", "]", "?"],
    "state_shape": (10, 32, 32),
    "map_shape": (14, 28),
    "ascii_to_tile": None,
}


class Game(metaclass=ABCMeta):
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        try:
            self.ascii = GameDescription[f'{name}_{version}']["ascii"]
            self.input_shape = GameDescription[f'{name}_{version}']["state_shape"]
            self.map_shape = GameDescription[f'{name}_{version}']["map_shape"]
        except:
            raise Exception(f'{name}_{version}' +
                            " data not implemented in env.py")
        if 'char_to_tile' in GameDescription[f'{name}_{version}']:
            self.char_to_tile = GameDescription[f'{name}_{version}']['char_to_tile']
        self.map_level = np.vectorize(lambda x: self.ascii[x])
        self.padding_index = 1

    def get_original_levels(self, path: str = None):
        if path is not None:
            file_pathes = glob(path + "/*")
            levels = []
            for f_name in file_pathes:
                with open(f_name, 'r') as f:
                    content = f.read()
                levels.append(content)
            return levels
        else:
            dir_path = os.path.join(
                gym_gvgai.dir, "envs", "games", f"{self.name}_{self.version}")
            file_pathes = glob(dir_path + "/*")
            levels = []
            for f_name in file_pathes:
                if f_name == os.path.join(
                        gym_gvgai.dir, "envs", "games", f"{self.name}_{self.version}", f"{self.name}.txt"):
                    continue
                with open(f_name, 'r') as f:
                    content = f.read()
                levels.append(content)
            return levels

    def level_str_to_tensor(self, level: str):
        level = level.split()
        level_numpy = np.zeros(self.input_shape)
        # padding
        level_numpy[self.padding_index, :, :] = 1
        # label : onehot vector of counts of map tile object.
        label_numpy = np.zeros(len(self.ascii))
        for i, s in enumerate(level):
            for j, c in enumerate(s):
                if c == "\n":
                    break
                level_numpy[self.padding_index, i, j] = 0
                level_numpy[self.ascii.index(c), i, j] = 1
                label_numpy[self.ascii.index(c)] += 1
        return torch.tensor(level_numpy), torch.tensor(label_numpy)

    def level_tensor_to_strs(self, tensor: torch.tensor):
        lvl_array = tensor[:, :, :self.map_shape[0],
                           :self.map_shape[1]].argmax(dim=1).cpu().numpy()
        lvls = self.map_level(lvl_array).tolist()
        lvl_strs = ["\n".join(["".join(row) for row in lvl]) for lvl in lvls]
        return lvl_strs

    @abstractmethod
    def get_features(self, level_str: str):
        '''
        diversity samplingのための特徴抽出
        '''
        pass

    @abstractmethod
    def evaluation(self, playable_levels: list[str]):
        '''
        学習完了後の評価
        '''
        pass

    @abstractmethod
    def check_playable(self, lvl_str: str):
        '''
        プレイアビリティのチェック
        '''
        pass

    @abstractmethod
    def check_similarity(self, level1: str, level2: str):
        '''
        ペアの類似度チェック
        '''
        pass
