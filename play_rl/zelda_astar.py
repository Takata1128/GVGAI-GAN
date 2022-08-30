import numpy as np
from PIL import Image
from gan.level_visualizer import LevelVisualizer
from play_rl.env import Env
from play_rl.wrappers import GridGame
from collections import deque


class AstarZelda():
    def __init__(self):
        self.actions = [3, 4, 2, 5]
        self.dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.replace_ascii = ['wall', 'withkey', 'nokey', 'monsterQuick',
                              'monsterSlow', 'monsterNormal', 'key', 'goal', 'sword', '']
        self.replace_char = ['w', 'A', 'A', 'e',
                             'e', 'e', '+', 'g', 'x', '.']

    def process(self, ascii_state: str):
        ascii_rows = ascii_state.split('\n')
        level_row = []

        for row in ascii_rows:
            l = row.split(',')
            s = ''
            for word in l:
                if self.replace_ascii.count(word):
                    id = self.replace_ascii.index(word)
                    s += self.replace_char[id]
                else:  # オブジェクト同士の重なり
                    if 'monster' in word:
                        s += 'e'
                    elif 'nokey' in word or 'withkey' in word:
                        s += 'A'
                    elif 'key' in word:
                        s += '+'
                    elif 'goal' in word:
                        s += 'g'
                    else:  # ???　
                        s += '?'
            level_row.append(s)
        return level_row

    # decide action
    def act(self, state: str):
        # print(state)
        g = self.process(state)
        # print(g)
        H = len(g)
        W = len(g[0])

        sx, sy = -1, -1
        gx, gy = -1, -1
        kx, ky = -1, -1

        for i in range(H):
            for j in range(W):
                if g[i][j] == "A":
                    sx, sy = i, j
                if g[i][j] == "g":
                    gx, gy = i, j
                if g[i][j] == "+":
                    kx, ky = i, j

        Q = deque()

        dist = [[-1] * W for _ in range(H)]

        # exist key
        # 鍵からの距離
        if (kx, ky) != (-1, -1):
            Q.append([kx, ky])
            dist[kx][ky] = 0
            while len(Q) > 0:
                x, y = Q.popleft()
                for dx, dy in self.dir:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < H and 0 <= ny < W):
                        continue
                    if g[nx][ny] == "w":
                        continue
                    if dist[nx][ny] == -1:
                        dist[nx][ny] = dist[x][y] + 1
                        Q.append([nx, ny])

        # not exist key
        # ゴールからの距離
        else:
            Q.append([gx, gy])
            dist[gx][gy] = 0
            while len(Q) > 0:
                x, y = Q.popleft()
                for dx, dy in self.dir:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < H and 0 <= ny < W):
                        continue
                    if g[nx][ny] == "w":
                        continue
                    if dist[nx][ny] == -1:
                        dist[nx][ny] = dist[x][y] + 1
                        Q.append([nx, ny])

        # sx,syの周囲４マスを調べて最も鍵/ゴールまでの距離が短くなるところに行く

        # print(dist)

        best = 1000000
        action = -1
        for i, (dx, dy) in enumerate(self.dir):
            nx, ny = sx+dx, sy+dy
            if (0 <= nx < H and 0 <= ny < W) and g[nx][ny] != 'w':
                if dist[nx][ny] < best:
                    action = self.actions[i]
                    best = dist[nx][ny]
        dir = self.dir[self.actions.index(action)]
        nx, ny = sx + dir[0], sy+dir[1]
        if g[nx][ny] == 'e':
            action = 1  # attack
        return action


def play_astar(level_str: str, env: GridGame, visualize: bool = False, env_def: Env = None):
    obs = env.set_level_reset(level_str=level_str)
    actor = AstarZelda()
    done = False
    step = 0
    obs, _, _, info = env.step(0)
    frames = None
    if visualize:
        level_visualizer = LevelVisualizer(env_def)
        frames = []

    while not done:
        if visualize:
            if info is None:
                p_level_img = np.array(level_visualizer.draw_level(level_str))
            else:
                p_level_img = np.array(
                    level_visualizer.draw_level_ascii(info['ascii']))
            image = Image.fromarray(p_level_img)
            frames.append(image)
        action = actor.act(
            state=info['ascii'])
        # print(action)
        obs, reward, done, info = env.step(action)
        step += 1
    return reward, step, frames
