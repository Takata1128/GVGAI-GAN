from __future__ import annotations
import numpy as np
from collections import deque
from .env import GameDescription


def tensor_to_level_str(name, tensor):
    ascii = GameDescription[name]["ascii"]
    map_level = np.vectorize(lambda x: ascii[x])

    lvl_array = tensor.argmax(dim=1).cpu().numpy()
    lvls = map_level(lvl_array).tolist()
    lvl_strs = ["\n".join(["".join(row) for row in lvl]) for lvl in lvls]
    return lvl_strs


HEIGHT = {'zelda_v0': 9, 'zelda_v1': 12, 'mario': 14}
WIDTH = {'zelda_v0': 13, 'zelda_v1': 16, 'mario': 28}


def check_playable(lvl_str: str, env_fullname: str):
    if env_fullname == 'mario':
        return check_playable_mario(lvl_str)
    elif env_fullname == 'zelda_v0':
        return check_playable_zelda(lvl_str, env_fullname)
    elif env_fullname == 'zelda_v1':
        return check_playable_zelda(lvl_str, env_fullname)
    else:
        raise NotImplementedError(f"Env {env_fullname} is not implemented!")


def check_level_similarity(lvl_str1: str, lvl_str2: str, env_name: str):
    if env_name == 'mario':
        return check_level_similarity_mario(lvl_str1, lvl_str2)
    elif env_name == 'zelda_v0' or env_name == 'zelda_v1':
        return check_level_similarity_zelda(lvl_str1, lvl_str2, env_name[-2:])
    else:
        raise NotImplementedError(f"Env {env_name} is not implemented!")


def check_playable_mario(lvl_str: str):
    pass


def check_playable_zelda(lvl_str: str, version: str = 'v1'):
    g = lvl_str.split()

    H = HEIGHT[version]
    W = WIDTH[version]

    sx, sy = -1, -1
    gx, gy = -1, -1
    kx, ky = -1, -1

    countA, countG, countK = 0, 0, 0
    ok = True
    for i in range(len(g)):
        for j in range(len(g[0])):
            if (i == 0 or i >= H - 1 or j == 0 or j >= W - 1) and g[i][j] != "w":
                ok = False
            if g[i][j] == "A":
                sx, sy = i, j
                countA += 1
            if g[i][j] == "g":
                gx, gy = i, j
                countG += 1
            if g[i][j] == "+":
                kx, ky = i, j
                countK += 1

    if not (countA == 1 and countG == 1 and countK == 1) or not ok:
        return False

    Q = deque()
    Q.append([sx, sy])

    dist = [[-1] * W for _ in range(H)]
    dist[sx][sy] = 0

    dir = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    while len(Q) > 0:
        x, y = Q.popleft()
        for dx, dy in dir:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if g[nx][ny] == "w":
                continue
            if dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                Q.append([nx, ny])

    return dist[gx][gy] != -1 and dist[kx][ky] != -1


def check_level_similarity_mario(level1, level2):
    n = 0
    hit = 0
    for i in range(HEIGHT['mario'], HEIGHT['mario'] * 2):
        for j in range(WIDTH['mario']):
            c1 = level1[i][j]
            c2 = level2[i][j]
            if c1 == "\n":
                continue
            if c1 == c2:
                hit += 1
            n += 1
    return hit / n


def check_level_similarity_zelda(level1: list[str], level2: list[str], version: str):
    n = 0
    hit = 0
    for i in range(HEIGHT[version]):
        for j in range(WIDTH[version]):
            c1 = level1[i][j]
            c2 = level2[i][j]
            if c1 == "\n":
                continue
            if c1 == c2:
                hit += 1
            n += 1
    return hit / n


def check_object_similarity(level1: list[str], level2: list[str], version: str):
    n = 0
    hit = 0

    for i in range(HEIGHT[version]):
        for j in range(WIDTH[version]):
            c1 = level1[i][j]
            c2 = level2[i][j]
            if c1 in ["\n", ".", "w", '1', '2', '3']:
                continue
            if c1 == c2:
                hit += 1
            n += 1
    return hit / n if n > 0 else None


def check_shape_similarity(level1: list[str], level2: list[str], version: str):
    n = 0
    hit = 0

    for i in range(HEIGHT[version]):
        for j in range(WIDTH[version]):
            c1 = level1[i][j]
            c2 = level2[i][j]
            if c1 == "\n" or c1 not in [".", "w"]:
                continue
            if c1 == c2:
                hit += 1
            n += 1
    return hit / n if n > 0 else None


def make_zelda_level(wall_p=0.20, height=12, width=16):
    requirements = ['+', 'A', 'g']
    requirements_positions = []

    for c in requirements:
        while True:
            rx = np.random.randint(1, width - 1)
            ry = np.random.randint(1, height - 1)
            pos = (rx, ry)
            if pos not in requirements_positions:
                requirements_positions.append(pos)
                break

    level_rows = [["." for i in range(width)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            if (i == 0 or i == height - 1 or j == 0 or j == width - 1):
                level_rows[i][j] = 'w'

            if np.random.rand() < wall_p:
                level_rows[i][j] = 'w'

            for k, pos in enumerate(requirements_positions):
                x, y = pos
                if (j, i) == pos:
                    level_rows[i][j] = requirements[k]

    level_rows = [''.join(row) for row in level_rows]
    ret = '\n'.join(level_rows)
    return ret
