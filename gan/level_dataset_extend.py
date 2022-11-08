import os
import shutil

from .game.env import Game
from .level_visualizer import GVGAILevelVisualizer, MarioLevelVisualizer
import numpy as np
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def state_to_level(state, mapping):
    layers = []
    for idx in mapping:
        layers.append(state[:, idx: idx + 1])
    x = torch.cat(layers, dim=1)
    return x


def make_another_aliens(lvl_str):
    another = ""
    lvl_str_rows = lvl_str.split("\n")
    for i in range(len(lvl_str_rows)):
        for j in range(len(lvl_str_rows[i])):
            c = lvl_str_rows[i][j]
            if np.random.rand() > 0.05:
                pass
            else:
                if c == ".":
                    c = "0"
            another += c
        if i < len(lvl_str_rows) - 1:
            another += "\n"
    return another


def make_another_zelda(lvl_str):
    another = ""
    enemy = ["1", "2", "3"]
    lvl_str_rows = lvl_str.split("\n")
    for i in range(len(lvl_str_rows)):
        for j in range(len(lvl_str_rows[i])):
            c = lvl_str_rows[i][j]
            if (
                i == 0
                or i == len(lvl_str_rows) - 1
                or j == 0
                or j == len(lvl_str_rows[i]) - 1
            ):
                pass
            else:
                if np.random.rand() > 0.05:
                    pass
                else:
                    if c == ".":
                        c = "w"
                    elif c == "w":
                        c = "."
                    elif c in enemy:
                        c = np.random.choice(enemy)
            another += c
        if i < len(lvl_str_rows) - 1:
            another += "\n"
    return another


def make_another_roguelike(lvl_str):
    another = ""
    lvl_str_rows = lvl_str.split("\n")
    for i in range(len(lvl_str_rows)):
        for j in range(len(lvl_str_rows[i])):
            c = lvl_str_rows[i][j]
            if (
                i == 0
                or i == len(lvl_str_rows) - 1
                or j == 0
                or j == len(lvl_str_rows[i]) - 1
            ):
                pass
            else:
                if np.random.rand() > 0.05:
                    pass
                else:
                    if c == ".":
                        c = "w"
                    elif c == "w":
                        c = "."
            another += c
        if i < len(lvl_str_rows) - 1:
            another += "\n"
    return another


make_another = {}
make_another['zelda'] = make_another_zelda
make_another['aliens'] = make_another_aliens
make_another['roguelike'] = make_another_roguelike


def prepare_dataset(game: Game, seed=0, extend_data=True, flip=True):
    np.random.seed(seed)
    if game.name == 'mario':
        visualizer = MarioLevelVisualizer(game, os.path.dirname(
            __file__) + f"/data/level/{game.name}_{game.version}/")
    else:
        visualizer = GVGAILevelVisualizer(game)

    train_dir_path = os.path.dirname(
        __file__) + f"/data/level/{game.name}_{game.version}/train/"

    # clean dirs
    if os.path.exists(train_dir_path):
        shutil.rmtree(train_dir_path)
    os.makedirs(train_dir_path)

    lvl_strs = visualizer.game.get_original_levels(
        f'/root/mnt/pcg/GVGAI-GAN/gan/data/level/{game.name}_{game.version}/originals')

    for j in range(len(lvl_strs)):
        index = j % len(lvl_strs)
        lvl_str = lvl_strs[index]
        lvl_str = lvl_str.split()
        target_shape = game.model_shape[-1]

        # PADDING
        for i in range(target_shape[0]):
            s = ''
            if i < len(lvl_str):
                s = lvl_str[i]
            width = len(s)
            for _ in range(target_shape[1] - width):
                s += game.ascii[1]
            if i < len(lvl_str):
                lvl_str[i] = s
            else:
                lvl_str.append(s)

        lvl_str = "\n".join(lvl_str)

        if extend_data:
            s = make_another[game.name](lvl_str)
        else:
            s = lvl_str
        with open(
            train_dir_path + f"{game.name}_{str(j)}", mode="w"
        ) as f:
            f.write(s)

    # for i, lvl_str in enumerate(lvl_strs):
    #     state_numpy = visualizer.game.level_str_to_ndarray(lvl_str)
    #     state_tensor = torch.unsqueeze(torch.Tensor(state_numpy), 0)
    #     states.append(state_tensor)
    #     # lvl_str_re = visualizer.game.level_tensor_to_strs(state_tensor)
    #     # with open(
    #     #     train_dir_path + f"{str(i)}.base", mode="w"
    #     # ) as f:
    #     #     f.write(lvl_str_re[0])
    #     if flip:
    #         states.append(torch.flip(state_tensor, [2]))
    #         states.append(torch.flip(state_tensor, [3]))
    #         states.append(torch.flip(state_tensor, [2, 3]))

    # for j in range(dataset_size):
    #     index = j % len(states)
    #     lvl_str_re = visualizer.game.level_tensor_to_strs(states[index])
    #     if extend_data:
    #         s = make_another[game_name](lvl_str_re[0])
    #     else:
    #         s = lvl_str_re[0]
    #     with open(
    #         train_dir_path + f"{game_name}_{str(j)}", mode="w"
    #     ) as f:
    #         f.write(s)


if __name__ == "__main__":
    prepare_dataset(game_name="aliens")
    prepare_dataset(game_name="zelda")
    prepare_dataset(game_name="roguelike")
