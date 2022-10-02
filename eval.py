from random import choice
import os
from tkinter import Grid
import gym
from play_rl.policy import Policy
from play_rl.wrappers import GridGame
from play_rl.zelda_astar import play_astar
from play_rl.env import Env
import torch
import numpy as np
from PIL import Image
from gan.level_visualizer import GVGAILevelVisualizer
import matplotlib.pyplot as plt


def play(level_str: str, env: GridGame, actor: Policy, visualize: bool = False, env_def: Env = None):
    obs = env.set_level_reset(level_str=level_str)
    done = False
    step = 0
    info = None
    rnn_hxs = torch.zeros((1, actor.recurrent_hidden_state_size))
    frames = None
    if visualize:
        level_visualizer = GVGAILevelVisualizer(env_def)
        frames = []

    while not done:
        obs = torch.FloatTensor(obs).unsqueeze(0)
        if visualize:
            if info is None:
                p_level_img = np.array(level_visualizer.draw_level(level_str))
            else:
                p_level_img = np.array(
                    level_visualizer.draw_level_ascii(info['ascii']))
            image = Image.fromarray(p_level_img)
            frames.append(image)
        _, action, _, rnn_hxs = actor.act(
            inputs=obs, rnn_hxs=rnn_hxs, masks=1.0, deterministic=True)
        obs, reward, done, info = env.step(action.detach().numpy())
        step += 1
    return reward, step, frames


def eval(level_str: str, env: GridGame, actor: Policy):
    reward, step = play(level_str=level_str, env=env, actor=actor)
    return reward


if __name__ == '__main__':
    game_name = 'zelda'
    device = torch.device('cpu')
    env_def = Env(name=game_name)
    level_visualizer = GVGAILevelVisualizer(env_def)
    env = GridGame(game_name=game_name, play_length=100,
                   shape=env_def.state_shape)

    def show(level_str):
        p_level_img = np.array(level_visualizer.draw_level(level_str))
        image = Image.fromarray(p_level_img)
        image.show()

    actor = Policy(obs_shape=env.observation_space.shape, action_space=env.action_space,
                   base_kwargs={"recurrent": True}).to(device=device)
    actor.load_state_dict(torch.load(
        "/root/mnt/GVGAI-GAN/play_rl/checkpoints/ppo/zelda_20220825102126.pt"))

    levels_dir = '/root/mnt/GVGAI-GAN/gan/data/level/zelda/generated/'

    for i in range(100):
        files = os.listdir(levels_dir)
        file = choice(files)
        print(file)
        with open(f'/root/mnt/GVGAI-GAN/gan/data/level/zelda/generated/{file}', mode='r') as f:
            level_str = f.read()
        show(level_str)
        reward, step, frames = play_astar(
            level_str=level_str, env=env, visualize=True, env_def=env_def)
