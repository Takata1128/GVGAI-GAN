import gym
from play_rl.wrappers import GridGame
from play_rl.policy import Policy
from gan.small_models import Generator
from gan.config import BaseConfig
from gan.env import Env
from gym import spaces
from __future__ import annotations
import torch

ACTION_LOW = -0.1
ACTION_HIGH = 0.1
ACTION_SHAPE = (32,)
OBSERVATION_SHAPE = (8, 12, 16)

ENV_LIMIT = 100


class GridGameOptimEnv(gym.Env):
    def __init__(self, gan_config: BaseConfig, env_def: Env, levels_dir_path: str, id: int):
        self.env_def = env_def
        self.gan_config = gan_config
        self.action_space = spaces.Box(
            ACTION_LOW, ACTION_HIGH, shape=ACTION_SHAPE)
        self.observation_space = spaces.Box(OBSERVATION_SHAPE)
        self.env = GridGame(
            env_def.name, 100, gan_config.input_shape, levels_dir_path, id=id)
        self.latent = torch.randn(gan_config.latent_size)
        self.generator = Generator(gan_config.input_shape[0], gan_config.model_shapes, (
            gan_config.latent_size,), gan_config.generator_filters, gan_config.use_self_attention_g, gan_config.use_conditional, gan_config.use_deconv_g)
        self.policy = Policy(
            self.env.observation_space.shape, self.env.action_space)
        self.steps = 0
        self.limit = ENV_LIMIT

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.latent += action
        state_tensor = self.generator(self.latent)
        level_str = self.env_def.level_tensor_to_strs(state_tensor)
        reward = self.reward(level_str)
        self.steps += 1
        done = (self.steps >= self.limit)
        return level_str[0], reward, done, None

    def reward(self, level_str: str):
        pass

    def close(self):
        self.env.close()
