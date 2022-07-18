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

    from collections import deque


def check_playable(lvl_str):
    g = lvl_str.split()
    H = len(g)
    W = len(g[0])

    sx, sy = -1, -1
    gx, gy = -1, -1
    kx, ky = -1, -1

    countA, countG, countK = 0, 0, 0
    ok = True
    for i in range(H):
        for j in range(W):
            if (i == 0 or i == H - 1 or j == 0 or j == W - 1) and g[i][j] != "w":
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


def check_level_similarity(level1: str, level2: str):
    n = 0
    hit = 0
    for c1, c2 in zip(level1, level2):
        if c1 == "\n":
            continue
        if c1 == c2:
            hit += 1
        n += 1
    return hit / n


def check_object_similarity(level1: str, level2: str):
    n = 0
    hit = 0
    for c1, c2 in zip(level1, level2):
        if c1 in ["\n", ".", "w"]:
            continue
        if c1 == c2:
            hit += 1
        n += 1
    return hit / n if n > 0 else None


def check_shape_similarity(level1: str, level2: str):
    n = 0
    hit = 0
    for c1, c2 in zip(level1, level2):
        if c1 == "\n" or c1 not in [".", "w"]:
            continue
        if c1 == c2:
            hit += 1
        n += 1
    return hit / n if n > 0 else None
