from tkinter import Grid
import gym
from play_rl.policy import Policy
from play_rl.wrappers import GridGame
import torch


def play(level_str: str, env: GridGame, actor: Policy):
    obs = env.set_level_reset(level_str=level_str)
    done = False
    step = 0
    rnn_hxs = torch.zeros((1, actor.recurrent_hidden_state_size))
    while not done:
        obs = torch.FloatTensor(obs).unsqueeze(0)
        _, action, _, rnn_hxs = actor.act(
            inputs=obs, rnn_hxs=rnn_hxs, masks=1.0)
        obs, reward, done, _ = env.step(action.detach().numpy())
        step += 1
    return reward, step


def eval(level_str: str, env: GridGame, actor: Policy):
    reward, step = play(level_str=level_str, env=env, actor=actor)
    return reward
