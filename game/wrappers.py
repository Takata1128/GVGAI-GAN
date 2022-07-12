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

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import pdb


def make_env(env_def, path, seed, rank, log_dir, allow_early_resets, **env_kwargs):
    def _thunk():
        if path:
            env = GridGame(
                env_def.name,
                env_def.length,
                env_def.state_shape,
                path,
                id=rank,
                **env_def.kwargs
            )
        else:
            env = GridGame(
                env_def.name,
                env_def.length,
                env_def.state_shape,
                id=rank,
                **env_def.kwargs
            )

        # env.seed(seed + rank)
        obs_shape = env.observation_space.shape

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets,
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
        make_env(env_def, level_path, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context="fork")
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #    if gamma is None:
    #        envs = VecNormalize(envs, ret=False)
    #    else:
    #        envs = VecNormalize(envs, gamma=gamma)

    envs = torch_env.VecPyTorch(envs, device)

    # if num_frame_stack is not None:
    #    envs = torch_env.VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #    envs = torch_env.VecPyTorchFrameStack(envs, 4, device)

    return envs


# Look at baseline wrappers and make a wrapper file: New vec_wrapper + game_wrapper
class GridGame(gym.Wrapper):
    def __init__(
        self,
        game,
        play_length,
        shape,
        path=None,
        id=0,
        reward_mode="time",
        reward_scale=2,
        elite_prob=0,
    ):
        """Returns Grid instead of pixels
        Sets the reward
        Generates new level on reset
        #PPO wants to maximize, Generator wants a score of 0
        --------
        """
        self.id = id
        self.name = game
        self.levels_dir = path

        self.level_id = -1
        self.version = 1
        self.env = gym_gvgai.make("gvgai-{}-lvl0-v{}".format(game, self.version))
        gym.Wrapper.__init__(self, self.env)

        self.compiles = False
        self.state = None
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

    def reset(self):
        self.steps = 0
        self.score = 0
        self.keyget = False
        level_ok_pathes = [
            p
            for p in os.listdir(os.path.join(self.levels_dir, "ok"))
            if os.path.splitext(p)[1] == ".txt"
        ]
        level_ng_pathes = [
            p
            for p in os.listdir(os.path.join(self.levels_dir, "ng"))
            if os.path.splitext(p)[1] == ".txt"
        ]
        if self.levels_dir is not None:
            if random.random() < self.elitep and level_ok_pathes:
                state = self.set_level(
                    os.path.join(self.levels_dir, "ok", random.choice(level_ok_pathes))
                )
            else:
                state = self.set_level(
                    os.path.join(self.levels_dir, "ng", random.choice(level_ng_pathes))
                )
        else:
            state = self.set_level()
        return state

    def step(self, action):
        if not isinstance(action, int):
            action = action.item()
        if not self.compiles:
            return self.state, -self.rscale * 2.0, True, {}
        _, r, done, info = self.env.step(action)

        # タイムアップ
        if self.steps >= self.play_length:
            done = True

        # プレイヤ消失
        if "nokey" not in info["ascii"] and "withkey" not in info["ascii"]:
            done = True

        # 報酬設計
        if self.rmode == "base":
            reward = self.get_reward(done, info["winner"], r)
        elif self.rmode == "time":
            reward = self.get_time_reward(done, info["winner"], r)
        elif self.rmode == "bonus":
            # 鍵をとったときボーナス
            keyget = False
            if not self.keyget and r == 1.0:
                self.keyget = True
                keyget = True
            reward = self.get_bonus_reward(done, info["winner"], keyget)
        elif self.rmode == None:
            reward = r
        else:
            raise Exception("Reward Scheme Not Implemented")

        state = self.get_state(info["grid"])
        self.steps += 1
        self.score += reward
        return state, reward, done, {"ascii": info["ascii"]}

    def get_bonus_reward(self, isOver, winner, keyget):
        reward = 0
        if isOver:
            if winner == "PLAYER_WINS":  # ゴール
                reward = self.rscale * 2.0
            elif winner == "PLAYER_LOSES":  # 敵にやられる
                reward = 0
            else:  # タイムアップ or 外に出る
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
        state = self.pad(grid)
        state = self.background(state)
        self.state = state.astype("float32")
        return state

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
            state = np.load(os.path.join(os.path.splitext(lvl)[0] + ".npy"))
            self.level_id = int(re.sub(r"\D", "", os.path.basename(lvl)))

            # if os.path.isfile(os.path.join(os.path.splitext(lvl)[0]) + ".no_compile"):
            #     self.compiles = False
            if "ng" in lvl:
                self.compiles = False
            else:
                try:
                    self.env.unwrapped._setLevel(lvl)
                    self.test_level()
                    self.compiles = True
                except Exception as e:
                    self.compiles = False
                    self.env.reset()
                except SystemExit:
                    # print("SystemExit")
                    self.compiles = False
                    self.restart(
                        "SystemExit", os.path.splitext(os.path.basename(lvl))[0]
                    )
        else:
            """既存ステージからランダムに生成"""
            self.compiles = True
            self.level_id = -1
            lvl = random.randint(0, 4)
            self.env.unwrapped._setLevel(lvl)
            self.env.reset()
            _, _, _, info = self.env.step(0)
            state = self.get_state(info["grid"])
        self.state = state
        return state

    def log(self, text):
        path = os.path.join(self.levels_dir, "log_{}".format(self.id))
        with open("log_{}".format(self.id), "a+") as log:
            log.write(str(text) + "\n")

    def log_reward(self, reward):
        if self.level_id >= 0:
            path = os.path.join(self.levels_dir, "rewards_{}.csv".format(self.id))
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

    def restart(self, e, path):
        # self.log(e)
        open(path + ".no_compile", "w").close()
        self.env = gym_gvgai.make("gvgai-{}-lvl0-v{}".format(self.name, self.version))


class CenteredGym(gym.Wrapper):
    def __init__(self, env, mapping, ascii):
        self.env = env
        self.name = self.env.name
        gym.Wrapper.__init__(self, self.env)

        self.avatar = mapping[ascii.index("A")]

        d, h, w = self.env.shape
        self.shape = (d, h + 2 * (h // 2), w + 2 * (w // 2))
        self.pad = (self.shape[1] // 2, self.shape[2] // 2)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.shape, dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        x, y = self.get_pos(obs)
        pad_dims = ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]))
        padded = np.pad(obs, pad_dims, mode="constant")
        centered = padded[:, y : y + self.shape[1], x : x + self.shape[2]]
        return centered

    def get_pos(self, obs):
        # map = obs[self.avatar]
        # pos = np.unravel_index(map.argmax(), map.shape)
        y, x = np.argwhere(obs.argmax(0) == self.avatar)[0]
        return (x, y)
