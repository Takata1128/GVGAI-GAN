import os
import shutil
from .level_visualizer import LevelVisualizer
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


def prepare_dataset(seed=0, extend_data=True, flip=True, clone_size=100, game_name="zelda", version='v1'):
    np.random.seed(seed)
    visualizer = LevelVisualizer(game_name, version=version)

    train_dir_path = os.path.dirname(
        __file__) + f"/data/level/{game_name}/train/"

    # clean dirs
    if os.path.exists(train_dir_path):
        shutil.rmtree(train_dir_path)
    os.makedirs(train_dir_path)

    lvl_strs = visualizer.game.get_original_levels()

    for i, lvl_str in enumerate(lvl_strs):
        state_numpy = visualizer.game.level_strs_to_ndarray(lvl_str)
        state_tensor = torch.unsqueeze(torch.Tensor(state_numpy), 0)
        states = []
        states.append(state_tensor)
        if flip:
            states.append(torch.flip(state_tensor, [2]))
            states.append(torch.flip(state_tensor, [3]))
            states.append(torch.flip(state_tensor, [2, 3]))
        for j, st in enumerate(states):
            lvl_str_re = visualizer.game.level_tensor_to_strs(st)
            img = visualizer.draw_level(lvl_str_re[0])
            img.save(os.path.dirname(
                __file__) + f"/data/level/{game_name}/"+f"level_{i}_{j}.jpg")
            for k in range(0, 250):
                lvl_str_re = visualizer.game.level_tensor_to_strs(st)
                if extend_data:
                    s = make_another[game_name](lvl_str_re[0])
                else:
                    s = lvl_str_re[0]
                with open(
                    train_dir_path + f"{game_name}_{str(i)}_{str(j)}_{str(k)}", mode="w"
                ) as f:
                    f.write(s)


if __name__ == "__main__":
    prepare_dataset(game_name="aliens")
    prepare_dataset(game_name="zelda")
    prepare_dataset(game_name="roguelike")
