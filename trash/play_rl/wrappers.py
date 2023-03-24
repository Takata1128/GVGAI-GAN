"""
Wrappers for VGDL Games
"""
import os
import csv
import random
import numpy as np
import re

# import timeout_decorator

import gym
import gym_gvgai

import a2c_ppo_acktr.envs as torch_env

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from .env import Env


def make_env(env_def, path, seed, rank, log_dir, allow_early_resets, **env_kwargs):
    def _thunk():
        env = GridGame(
            game_name=env_def.name,
            play_length=100,
            shape=env_def.state_shape,
            levels_dir_path=path,
            id=rank
        )
        return env
    return _thunk


def make_vec_envs(
    env_def,
    level_path,
    seed,
    num_processes,
    log_dir,
    device,
    allow_early_resets,
    num_frame_stack=None,
):
    envs = [
        make_env(env_def=env_def, path=level_path, seed=seed, rank=i,
                 log_dir=log_dir, allow_early_resets=allow_early_resets)
        for i in range(num_processes)
    ]
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    envs = torch_env.VecPyTorch(envs, device)
    return envs


# Look at baseline wrappers and make a wrapper file: New vec_wrapper + game_wrapper
class GridGame(gym.Wrapper):
    def __init__(
        self,
        game_name,
        play_length,
        shape,
        levels_dir_path=None,
        id=0,
        reward_mode="bonus",
        reward_scale=1.0,
        elite_prob=0,
    ):
        self.id = id
        self.name = game_name
        self.setting = Env(self.name)
        self.levels_dir = levels_dir_path

        self.level_id = -1
        self.version = 1
        self.env = gym_gvgai.make(
            "gvgai-{}-lvl0-v{}".format(game_name, self.version))
        gym.Wrapper.__init__(self, self.env)

        self.steps = 0
        self.score = 0
        self.play_length = play_length
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )

        self.keyget = False  # 鍵をとったかどうか
        self.elitep = elite_prob
        self.rmode = reward_mode
        self.rscale = reward_scale

        self.level_pathes = os.listdir(self.levels_dir)

    def reset(self):
        self.steps = 0
        self.score = 0
        self.keyget = False
        self.env.reset()
        if self.levels_dir is not None:
            state = self.set_level(
                os.path.join(self.levels_dir,
                             np.random.choice(self.level_pathes))
            )
        else:
            state = self.set_level()
        return state

    def set_level_reset(self, level_str: str):
        self.steps = 0
        self.score = 0
        self.keyget = False
        self.env.reset()
        level_path = f'/var/tmp/{self.id}'
        with open(level_path, mode='w') as f:
            f.write(level_str)
        state = self.set_level(level_path)
        return state

    def step(self, action):
        if not isinstance(action, int):
            action = action.item()
        _, r, done, info = self.env.step(action)

        # タイムアップ
        if self.steps >= self.play_length:
            done = True

        # ステップのペナルティ
        reward = -0.01

        # 報酬設計
        if self.rmode == "base":
            reward += self.get_reward(done, info["winner"], r)
        elif self.rmode == "time":
            reward += self.get_time_reward(done, info["winner"], r)
        elif self.rmode == "bonus":
            # 鍵をとったときボーナス
            keyget = False
            if not self.keyget and r == 1.0:
                self.keyget = True
                keyget = True
            reward += self.get_bonus_reward(done, info["winner"], keyget)
        elif self.rmode == None:
            reward = r
        else:
            raise Exception("Reward Scheme Not Implemented")

        state = self.get_state(info["grid"])
        self.steps += 1
        self.score += reward
        if "episode" in info.keys():
            info["episode"]["r"] = self.score
        return state, reward, done, info

    def get_bonus_reward(self, isOver, winner, keyget):
        reward = 0
        if isOver:
            if winner == "PLAYER_WINS":  # ゴール
                reward = self.rscale * 2.0
            elif winner == "PLAYER_LOSES":  # 敵にやられる
                reward = 0
            else:  # タイムアップ
                reward = -self.rscale
        if keyget:  # 鍵GET
            reward = self.rscale
        return reward

    def get_time_reward(self, isOver, winner, r):
        if isOver:
            if winner == "PLAYER_WINS":
                reward = self.rscale - self.steps / self.play_length
            else:
                reward = -self.rscale + self.steps / self.play_length
            self.log_reward(self.score + reward)
            return reward
        else:
            if r > 0:
                return 1 / self.play_length
            else:
                return 0

    def get_reward(self, isOver, winner, r):
        if isOver:
            if winner == "PLAYER_WINS":
                reward = 1
            else:
                reward = -1
            self.log_reward(self.score + reward)
            return reward
        else:
            return 0

    def get_state(self, grid):
        # state = self.pad(grid)
        # state = self.background(state)
        return grid.astype("float32")

    def set_level(self, lvl=None):
        if lvl is not None and isinstance(lvl, int):
            """既存ステージ指定して生成"""
            self.compiles = True
            self.level_id = -1
            self.env.unwrapped._setLevel(lvl)
            self.env.reset()
            _, _, _, info = self.env.step(0)
            state = self.get_state(info["grid"])
        elif lvl:
            """テキストファイルからレベル生成"""
            with open(lvl) as f:
                lvl_str = f.read()
            try:
                self.env.unwrapped._setLevel(lvl)
                self.test_level()
            except Exception as e:
                self.env.reset()
            except SystemExit:
                print("SystemExit")
                self.restart(
                    "SystemExit", os.path.splitext(os.path.basename(lvl))[0]
                )
            s, _, _, info = self.env.step(0)
            state = self.get_state(info["grid"])
            self.level_id = int(re.sub(r"\D", "", os.path.basename(lvl)))
        else:
            """既存ステージからランダムに生成"""
            self.compiles = True
            self.level_id = -1
            lvl = random.randint(0, 4)
            self.env.unwrapped._setLevel(lvl)
            self.env.reset()
            _, _, _, info = self.env.step(0)
            state = self.get_state(info["grid"])
        return state

    def log(self, text):
        path = os.path.join(self.levels_dir, "log_{}".format(self.id))
        with open("log_{}".format(self.id), "a+") as log:
            log.write(str(text) + "\n")

    def log_reward(self, reward):
        if self.level_id >= 0:
            path = os.path.join(
                self.levels_dir, "rewards_{}.csv".format(self.id))
            add_header = not os.path.exists(path)
            with open(path, "a+") as rewards:
                writer = csv.writer(rewards)
                if add_header:
                    writer.writerow(["level", "reward"])
                writer.writerow((self.level_id, reward))

    # @timeout_decorator.timeout(1, use_signals=False)
    def test_level(self):
        self.env.reset()
        self.env.step(0)
        self.env.reset()

    def pad(self, state):
        pad_h = max(self.shape[-2] - state.shape[-2], 0)
        pad_w = max(self.shape[-1] - state.shape[-1], 0)
        pad_height = (pad_h // 2, pad_h - pad_h // 2)
        pad_width = (pad_w // 2, pad_w - pad_w // 2)
        padding = ((0, 0), pad_height, pad_width)
        return np.pad(state, padding, "constant")

    def background(self, state):
        background = 1 - np.clip(np.sum(state, 0), 0, 1)
        background = np.expand_dims(background, 0)
        return np.concatenate([state, background])

    def mapping(self, state):
        ret = np.zeros(self.setting.state_shape)
        for i, e in enumerate(self.setting.mapping):
            ret[i] = state[e]
        return ret

    def restart(self, e, path):
        # self.log(e)
        open(path + ".no_compile", "w").close()
        self.env = gym_gvgai.make(
            "gvgai-{}-lvl0-v{}".format(self.name, self.version))
