from __future__ import annotations
import numpy as np
from collections import deque
from .env import GameDescription, Env
from .config import BaseConfig
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# def check_property(property, prop_dict, env_fullname):
#     if env_fullname == 'zelda_v1':
#         return check_property_zelda(property, prop_dict)
#     else:
#         raise NotImplementedError()


# def check_property_zelda(property, prop_dict):
#     count = prop_dict[property] if property in prop_dict else 0
#     if count < int(len(dataset_files) * config.bootstrap_property_filter):


def get_property(level: str, env_fullname: str):
    if env_fullname == "zelda_v1":
        return get_property_zelda(level)
    elif env_fullname == 'roguelike_v0':
        return get_property_rogue(level)
    elif env_fullname == 'mario_v0':
        return get_property_mario(level)
    else:
        raise NotImplementedError()


def get_property_mario(level: str):
    return (0, 0, 0)


def get_property_rogue(level: str):
    index_agent, index_key, index_goal = 0, 0, 0
    for i, c in enumerate(level):
        if c == 'A':
            index_agent = i
        elif c == 'k':
            index_key = i
        elif c == 'x':
            index_goal = i
    return (index_agent, index_key, index_goal)


def get_property_zelda(level: str):
    index_agent, index_key, index_goal = 0, 0, 0
    for i, c in enumerate(level):
        if c == 'A':
            index_agent = i
        elif c == '+':
            index_key = i
        elif c == 'g':
            index_goal = i
    return (index_agent, index_key, index_goal)


def evaluation(playable_levels: list[str], env_fullname: str):
    if env_fullname == "zelda_v1":
        return evaluation_zelda(playable_levels)
    if env_fullname == 'mario_v0':
        return evaluation_mario(playable_levels)
    else:
        raise NotImplementedError()


def evaluation_mario(playable_levels: list[str]):
    pass


def evaluation_zelda(playable_levels: list[str]):
    def check_level_hamming(level1: str, level2: str):
        hit = 0
        for c1, c2 in zip(level1, level2):
            if c1 == "\n":
                continue
            if c1 != c2:
                hit += 1
        return hit

    def check_level_object_duprecated(level1: str, level2: str):
        k, g, a, all = 0, 0, 0, 0
        for c1, c2 in zip(level1, level2):
            if c1 == "\n":
                continue
            if c1 == 'g' and c1 == c2:
                g = 1
            if c1 == '+' and c1 == c2:
                k = 1
            if c1 == 'A' and c1 == c2:
                a = 1
        if k and g and a:
            all = 1
        return k, g, a, all

    key_duplication, goal_duplication, player_duplication, total_object_duplication, total_hamming_dist, dup90, n = 0, 0, 0, 0, 0, 0, 0
    levels_small_set = playable_levels[:1000]
    for i in range(len(levels_small_set)):
        for j in range(len(levels_small_set)):
            if i < j:
                continue
            kd, gd, pd, sod = check_level_object_duprecated(
                levels_small_set[i], levels_small_set[j])
            key_duplication += kd
            goal_duplication += gd
            player_duplication += pd
            total_object_duplication += sod
            hamming = check_level_hamming(
                levels_small_set[i], levels_small_set[j])
            total_hamming_dist += hamming
            if hamming / (12 * 16) <= 0.10:
                dup90 += 1
            n += 1

    unique_levels = list(set(playable_levels))

    metrics = {}
    metrics["Final Duplication Ratio"] = 1 - \
        len(unique_levels) / len(playable_levels)
    metrics["Hamming Distance"] = total_hamming_dist / n
    metrics["90%% duplication Rate"] = dup90 / n
    metrics["Object Duplication Ratio"] = total_object_duplication / n
    metrics["Key Duplication Ratio"] = key_duplication / n
    metrics["Goal Duplication Ratio"] = goal_duplication / n
    metrics["Player Duplication Ratio"] = player_duplication / n
    print("Duplication Ratio:", 1 - len(unique_levels) / len(playable_levels))
    print("Hamming Distance:", total_hamming_dist / n)
    print("Obj Duplication ratio:", total_object_duplication / n)

    n_w, n_f, n_e, n = 0, 0, 0, 0
    vw, vf, ve = [], [], []

    for level in playable_levels:
        w = 0
        f = 0
        e = 0
        n += 1
        for c in level:
            if c == 'w':
                w += 1
                n_w += 1
            elif c == '.':
                f += 1
                n_f += 1
            elif c in ['1', '2', '3']:
                e += 1
                n_e += 1
        vw.append(w)
        vf.append(f)
        ve.append(e)
    sw, sf, se = 0, 0, 0
    for nw, nf, ne in zip(vw, vf, ve):
        sw += (nw - n_w / n)**2
        sf += (nf - n_f / n)**2
        se += (ne - n_e / n)**2

    metrics["Wall avg."] = n_w / n
    metrics["Floor avg."] = n_f / n
    metrics["Enemy avg."] = n_e / n
    metrics["Wall var."] = sw / n
    metrics["Floor var."] = sf / n
    metrics["Enemy var."] = se / n

    print('Ave. Wall:', n_w / n)
    print('Ave. Floor:', n_f / n)
    print('Ave. Enemy:', n_e / n)
    print('Std. Wall:', sw / n)
    print('Std. Floor:', sf / n)
    print('Std. Enemy:', se / n)

    return metrics


def evaluation_rogue(playable_levels: list[str]):
    def check_level_hamming(level1: str, level2: str):
        hit = 0
        for c1, c2 in zip(level1, level2):
            if c1 == "\n":
                continue
            if c1 != c2:
                hit += 1
        return hit

    def check_level_object_duprecated(level1: str, level2: str):
        k, g, a, all = 0, 0, 0, 0
        for c1, c2 in zip(level1, level2):
            if c1 == "\n":
                continue
            if c1 == 'x' and c1 == c2:
                g = 1
            if c1 == 'k' and c1 == c2:
                k = 1
            if c1 == 'A' and c1 == c2:
                a = 1
        if k and g and a:
            all = 1
        return k, g, a, all

    key_duplication, goal_duplication, player_duplication, total_object_duplication, total_hamming_dist, n = 0, 0, 0, 0, 0, 0
    levels_small_set = playable_levels[:1000]
    for i in range(len(levels_small_set)):
        for j in range(len(levels_small_set)):
            if i == j:
                continue
            kd, gd, pd, sod = check_level_object_duprecated(
                levels_small_set[i], levels_small_set[j])
            key_duplication += kd
            goal_duplication += gd
            player_duplication += pd
            total_object_duplication += sod
            total_hamming_dist += check_level_hamming(
                levels_small_set[i], levels_small_set[j])
            n += 1

    unique_levels = list(set(playable_levels))

    metrics = {}
    metrics["Final Duplication Ratio"] = 1 - \
        len(unique_levels) / len(playable_levels)
    metrics["Hamming Distance"] = total_hamming_dist / n
    metrics["Object Duplication Ratio"] = total_object_duplication / n
    metrics["Key Duplication Ratio"] = key_duplication / n
    metrics["Goal Duplication Ratio"] = goal_duplication / n
    metrics["Player Duplication Ratio"] = player_duplication / n
    print("Duplication Ratio:", 1 - len(unique_levels) / len(playable_levels))
    print("Hamming Distance:", total_hamming_dist / n)
    print("Obj Duplication ratio:", total_object_duplication / n)

    n_w, n_f, n_e, n = 0, 0, 0, 0
    vw, vf, ve = [], [], []

    for level in playable_levels:
        w = 0
        f = 0
        e = 0
        n += 1
        for c in level:
            if c == 'w':
                w += 1
                n_w += 1
            elif c == '.':
                f += 1
                n_f += 1
            elif c in ['r', 'p']:
                e += 1
                n_e += 1
        vw.append(w)
        vf.append(f)
        ve.append(e)
    sw, sf, se = 0, 0, 0
    for nw, nf, ne in zip(vw, vf, ve):
        sw += (nw - n_w / n)**2
        sf += (nf - n_f / n)**2
        se += (ne - n_e / n)**2

    metrics["Wall avg."] = n_w / n
    metrics["Floor avg."] = n_f / n
    metrics["Enemy avg."] = n_e / n
    metrics["Wall var."] = sw / n
    metrics["Floor var."] = sf / n
    metrics["Enemy var."] = se / n

    print('Ave. Wall:', n_w / n)
    print('Ave. Floor:', n_f / n)
    print('Ave. Enemy:', n_e / n)
    print('Std. Wall:', sw / n)
    print('Std. Floor:', sf / n)
    print('Std. Enemy:', se / n)

    return metrics


def kmeans_select(unique_playable_levels: list[str], config: BaseConfig, env: Env):
    def level_str_to_features(level_str):
        level_str = level_str.split()
        ret = np.zeros((len(level_str), len(level_str[0])))
        for i, s in enumerate(level_str):
            for j, c in enumerate(level_str[i]):
                ret[i, j] = env.ascii.index(
                    c) / len(env.ascii)
        return ret.reshape(-1)

    def elbow(levels_reduced):
        prev_sse = -1
        now_max = 0
        elbow = 1
        for n_cluster in range(1, min(config.bootstrap_max_count, len(levels_reduced))):
            kmeans = KMeans(n_clusters=n_cluster, random_state=0)
            kmeans.fit(levels_reduced)
            sse = kmeans.inertia_
            if prev_sse > 0:
                if abs(sse - prev_sse) > now_max:
                    now_max = abs(sse - prev_sse)
                    elbow = n_cluster
            prev_sse = sse
        return elbow

    # levels -> feature vectors
    playable_levels_numpy = np.array(
        list(map(level_str_to_features, unique_playable_levels)))
    # 2D PCA
    pca = PCA(n_components=2)
    levels_reduced = pca.fit_transform(playable_levels_numpy)

    # k-means with elbow
    kmeans = KMeans(n_clusters=elbow(levels_reduced), random_state=0)
    kmeans.fit(levels_reduced)
    indices = []

    # correct nearest centers
    for center in kmeans.cluster_centers_:
        dist = 1000
        index = -1
        for i, lr in enumerate(levels_reduced):
            dist_tmp = (center[0] - lr[0])**2 + (center[1] - lr[1])**2
            if dist_tmp < dist:
                dist = dist_tmp
                index = i
        indices.append(index)

    # result
    result_levels = []
    for index in indices:
        result_levels.append(unique_playable_levels[index])

    return result_levels


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
    if env_fullname == 'mario_v0':
        return check_playable_mario(lvl_str)
    elif env_fullname == 'zelda_v0':
        return check_playable_zelda(lvl_str, env_fullname)
    elif env_fullname == 'zelda_v1':
        return check_playable_zelda(lvl_str, env_fullname)
    elif env_fullname == 'roguelike_v0':
        return check_playable_rogue(lvl_str)
    elif env_fullname == 'boulderdash_v0':
        return check_playable_boulderdash(lvl_str)
    else:
        raise NotImplementedError(f"Env {env_fullname} is not implemented!")


def check_level_similarity(lvl_str1: str, lvl_str2: str, env_fullname: str):
    if env_fullname == 'mario_v0':
        return check_level_similarity_mario(lvl_str1, lvl_str2)
    elif env_fullname == 'zelda_v0' or env_fullname == 'zelda_v1':
        return check_level_similarity_zelda(lvl_str1, lvl_str2, env_fullname)
    elif env_fullname == 'roguelike_v0':
        return check_level_similarity_rogue(lvl_str1, lvl_str2)
    else:
        raise NotImplementedError(f"Env {env_fullname} is not implemented!")


def check_playable_mario(lvl_str: str):
    g = lvl_str.split()
    ok = True
    for row in g[:14]:
        for c in row:
            pass
    mlen = 0
    len = 0
    for c in g[-1]:
        if c == '-':
            len += 1
            mlen = max(len, mlen)
        else:
            mlen = 0
    if mlen > 5:
        ok = False
    return ok


def check_playable_boulderdash(lvl_str: str):
    pass


def check_playable_rogue(lvl_str: str):
    g = lvl_str.split()

    H = 21
    W = 22

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
            if g[i][j] == "x":
                gx, gy = i, j
                countG += 1
            if g[i][j] == "k":
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


def check_level_similarity_zelda(level1: list[str], level2: list[str], env_fullname: str):
    n = 0
    hit = 0
    for i in range(HEIGHT[env_fullname]):
        for j in range(WIDTH[env_fullname]):
            c1 = level1[i][j]
            c2 = level2[i][j]
            if c1 == "\n":
                continue
            if c1 == c2:
                hit += 1
            n += 1
    return hit / n


def check_level_similarity_rogue(level1: list[str], level2: list[str]):
    n = 0
    hit = 0
    for i in range(21):
        for j in range(22):
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
