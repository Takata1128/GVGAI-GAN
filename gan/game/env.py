from __future__ import annotations
from glob import glob
from abc import ABCMeta, abstractmethod
from unittest.mock import NonCallableMagicMock
import gym_gvgai
import torch
import numpy as np
import os

GameDescription = {}
GameDescription["aliens"] = {
    "ascii": [".", "0", "1", "2", "A"],
    'ascii_to_tile': {
        "background": ["background"],
        "base": ["background", "base"],
        "portalSlow": ["background", "portalSlow"],
        "portalFast": ["background", "portalFast"],
        "avatar": ["background", "avatar"]
    },
    "state_shape": (5, 12, 32),
    "model_shape": [(3, 4), (6, 8), (12, 16), (12, 32)],
}
GameDescription["zelda_v1"] = {
    "ascii": [".", "w", "g", "+", "1", "2", "3", "A"],
    "ascii_to_tile": {
        "": ["floor"],
        "wall": ["wall"],
        "goal": ["floor", "goal"],
        "key": ["floor", "key"],
        "sword": ["floor", "sword"],
        "monsterQuick": ["floor", "monsterQuick"],
        "monsterNormal": ["floor", "monsterNormal"],
        "monsterSlow": ["floor", "monsterSlow"],
        "nokey": ["floor", "nokey"],
        "withkey": ["floor", "withkey"],
    },
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
    "model_shape": [(3, 4), (6, 8), (12, 16)],
}
GameDescription["zelda_v0"] = {
    "ascii": [".", "w", "g", "+", "1", "2", "3", "A"],
    "ascii_to_tile": {
        "": ["floor"],
        "wall": ["wall"],
        "goal": ["floor", "goal"],
        "key": ["floor", "key"],
        "sword": ["floor", "sword"],
        "monsterQuick": ["floor", "monsterQuick"],
        "monsterNormal": ["floor", "monsterNormal"],
        "monsterSlow": ["floor", "monsterSlow"],
        "nokey": ["floor", "nokey"],
        "withkey": ["floor", "withkey"],
    },
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
    "model_shape": [(3, 4), (6, 8), (12, 16)],
}
GameDescription["mario_v0"] = {
    "ascii": ["X", "S", '-', "Q", "E", "<", ">", "[", "]", "?"],
    "state_shape": (10, 32, 32),
    "map_shape": (14, 28),
    "model_shape": [(4, 4), (8, 8), (16, 16), (32, 32)],
    "ascii_to_tile": None,
}

GameDescription["roguelike"] = {
    "ascii": [".", 'w', "x", "s", "g", "r", "p", "h", "k", 'l', 'm', 'A'],
    "ascii_to_tile": {
        "": ["floor"],
        'wall': ['wall'],
        "exit": ["floor", "exit"],
        "weapon": ["floor", 'weapon'],
        "gold": ["floor", 'gold'],
        "spider": ["floor", "spider"],
        "phantom": ["floor", "phantom"],
        "health": ["floor", "health"],
        "key": ["floor", "key"],
        "lock": ["floor", "lock"],
        "market": ["floor", "market"],
        "avatar": ["floor", "avatar"],
    },
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
    "state_shape": (12, 24, 24),
    "map_shape": (22, 23),
    "model_shape": [(6, 6), (12, 12), (24, 24)],
}


map_shapes = {
    'mario_v0': [14, 28],
    'zelda_v0': [9, 13],
    'zelda_v1': [12, 16],
    'rogue_v0': [22, 23],
}


class Game(metaclass=ABCMeta):
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        try:
            self.ascii = GameDescription[f'{name}_{version}']["ascii"]
            self.input_shape = GameDescription[f'{name}_{version}']["state_shape"]
            self.map_shape = GameDescription[f'{name}_{version}']["map_shape"]
            self.model_shape = GameDescription[f'{name}_{version}']["model_shape"]
            self.ascii_to_tile = GameDescription[f'{name}_{version}']["ascii_to_tile"]
        except:
            raise Exception(f'{name}_{version}' +
                            " data not implemented in env.py")
        if 'char_to_tile' in GameDescription[f'{name}_{version}']:
            self.char_to_tile = GameDescription[f'{name}_{version}']['char_to_tile']
        self.map_level = np.vectorize(lambda x: self.ascii[x])

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

    def level_str_to_ndarray(self, lvl_str: str):
        ret = np.zeros(
            (len(self.ascii),
             self.input_shape[1], self.input_shape[2]),
        )
        index = 0
        for i, c in enumerate(lvl_str):
            if c == "\n":
                continue
            ret[self.ascii.index(c), index // self.input_shape[2],
                index % self.input_shape[2]] = 1
            index += 1
        return ret

    def level_tensor_to_strs(self, tensor: torch.tensor):
        lvl_array = tensor.argmax(dim=1).cpu().numpy()
        lvls = self.map_level(lvl_array).tolist()
        lvl_strs = ["\n".join(["".join(row) for row in lvl]) for lvl in lvls]
        return lvl_strs

    @abstractmethod
    def get_property(self, level_str: str):
        pass

    @abstractmethod
    def evaluation(self, playable_levels: list[str]):
        pass

    @abstractmethod
    def check_playable(self, lvl_str: str):
        pass

    @abstractmethod
    def check_similarity(self, level1: str, level2: str):
        pass
