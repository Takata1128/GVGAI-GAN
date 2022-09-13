from __future__ import annotations
from glob import glob
from unittest.mock import NonCallableMagicMock
import gym_gvgai
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
    "requirements": ["A"],
}
GameDescription["zelda"] = {
    "ascii": [".", "w", "g", "+", "1", "2", "3", "A"],
    "ascii_to_tile": {
        "goal": ["floor", "goal"],
        "": ["floor"],
        "floor": ["floor"],
        "key": ["floor", "key"],
        "nokey": ["floor", "nokey"],
        "withkey": ["floor", "withkey"],
        "sword": ["floor", "sword"],
        "monsterQuick": ["floor", "monsterQuick"],
        "monsterNormal": ["floor", "monsterNormal"],
        "monsterSlow": ["floor", "monsterSlow"],
        "wall": ["wall"],
    },
    "mapping": [13, 0, 3, 4, 10, 11, 12, 7],
    "state_shape": (8, 12, 16),
    "model_shape": [(3, 4), (6, 8), (12, 16)],
    "requirements": ["A", "g", "+"],
}

GameDescription["roguelike"] = {
    "ascii": [".", 'w', "x", "s", "g", "r", "p", "h", "k", 'l', 'm', 'A'],
    "ascii_to_tile": {
        "": ["floor"],
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
        'wall': ['wall']
    },
    "state_shape": (12, 21, 22),
    "model_shape": [(5, 5), (10, 11), (21, 22)],
    "requirements": ["x", 'A'],
}


class Env:
    def __init__(self, name, version='v1'):
        self.name = name
        self.version = version
        try:
            self.ascii = GameDescription[name]["ascii"]
            self.mapping = GameDescription[name]['mapping']
            self.state_shape = GameDescription[name]["state_shape"]
            self.model_shape = GameDescription[name]["model_shape"]
            self.requirements = GameDescription[name]["requirements"]
            self.ascii_to_tile = GameDescription[name]["ascii_to_tile"]
        except:
            raise Exception(name + " data not implemented in env.py")

        self.map_level = np.vectorize(lambda x: self.ascii[x])

    def get_original_levels(self, path=None):
        if path is not None:
            file_pathes = glob(path+"/*")
            levels = []
            for f_name in file_pathes:
                with open(f_name, 'r') as f:
                    content = f.read()
                levels.append(content)
            return levels
        else:
            dir_path = os.path.join(
                gym_gvgai.dir, "envs", "games", f"{self.name}_{self.version}")
            file_pathes = glob(dir_path+"/*")
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
             self.state_shape[1], self.state_shape[2]),
        )
        index = 0
        for i, c in enumerate(lvl_str):
            if c == "\n":
                continue
            ret[self.ascii.index(c), index//self.state_shape[2],
                index % self.state_shape[2]] = 1
            index += 1
        return ret

    def level_tensor_to_strs(self, tensor):
        lvl_array = tensor.argmax(dim=1).cpu().numpy()
        lvls = self.map_level(lvl_array).tolist()
        lvl_strs = ["\n".join(["".join(row) for row in lvl]) for lvl in lvls]
        return lvl_strs

    def pass_requirements(self, lvl_str):
        num_ok = sum(lvl_str.count(i) >= 1 for i in self.requirements)
        return num_ok == len(self.requirements)
