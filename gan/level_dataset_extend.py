import shutil
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.env import Env
from game.wrappers import GridGame
from level_visualizer import LevelVisualizer
from utils import check_level_similarity
import matplotlib.pyplot as plt
import torch
import numpy as np


def state_to_level(state, mapping):
    layers = []
    for idx in mapping:
        layers.append(state[:, idx : idx + 1])
    x = torch.cat(layers, dim=1)
    return x


def make_another(lvl_str):
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


def prepare_dataset(seed=0, extend_data=True, flip=True, game_name="zelda"):
    np.random.seed(seed)

    reward_mode = "time"
    reward_scale = 1.0
    elite_prob = 0.5
    env_def = Env(
        game_name,
        wrapper_args={
            "reward_mode": reward_mode,
            "reward_scale": reward_scale,
            "elite_prob": elite_prob,
        },
    )

    env = GridGame("zelda", 1000, env_def.state_shape)
    visualizer = LevelVisualizer("zelda")

    testpath = os.path.dirname(__file__) + "/data/level/test/level_"
    trainpath = os.path.dirname(__file__) + "/data/level/train/level_"

    # clean dirs
    shutil.rmtree(os.path.dirname(__file__) + "/data/level/train/")
    os.makedirs(os.path.dirname(__file__) + "/data/level/train/")

    for i in range(0, 5):
        state = env.set_level(i)
        state = torch.unsqueeze(torch.Tensor(state), 0)
        state = state_to_level(state, env_def.mapping)
        states = []
        states.append(state)
        if flip:
            states.append(torch.flip(state, [2]))
            states.append(torch.flip(state, [3]))
            states.append(torch.flip(state, [2, 3]))
        for j, st in enumerate(states):
            lvl_str = env_def.create_levels(st)
            with open(testpath + str(i) + "_" + str(j), mode="w") as f:
                f.write(lvl_str[0])
            img = visualizer.draw_level(lvl_str[0])
            plt.imshow(img)
            plt.show()
            for k in range(0, 250):
                lvl_str = env_def.create_levels(st)
                if extend_data:
                    s = make_another(lvl_str[0])
                else:
                    s = lvl_str[0]
                with open(
                    trainpath + str(i) + "_" + str(j) + "_" + str(k), mode="w"
                ) as f:
                    f.write(s)


if __name__ == "__main__":
    reward_mode = "time"
    reward_scale = 1.0
    elite_prob = 0.5
    env_def = Env(
        "zelda",
        wrapper_args={
            "reward_mode": reward_mode,
            "reward_scale": reward_scale,
            "elite_prob": elite_prob,
        },
    )

    env = GridGame("zelda", 1000, env_def.state_shape)

    score = 0
    count = 0
    testpath = os.path.dirname(__file__) + "/data/level/test/level_"
    for i in range(5):
        sti = env.set_level(i)
        sti = state_to_level(torch.unsqueeze(torch.Tensor(sti), 0), env_def.mapping)
        sti_str = env_def.create_levels(sti)
        with open(testpath + str(i) + "n", mode="w") as f:
            f.write(sti_str[0])
        for j in range(i + 1, 5):
            stj = env.set_level(j)
            stj = state_to_level(torch.unsqueeze(torch.Tensor(stj), 0), env_def.mapping)
            stj_str = env_def.create_levels(stj)
            score += check_level_similarity(sti_str[0], stj_str[0])
            print(score)
            count += 1

    print(score / count)
