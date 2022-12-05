from __future__ import annotations
from .env import Game
from collections import deque


class Zelda(Game):
    def __init__(self, name, version):
        super().__init__(name, version)
        if version == 'v0':
            self.height = 9
            self.width = 13
        else:
            self.height = 12
            self.width = 16

    def check_playable(self, lvl_str: str):
        g = lvl_str.split()
        sx, sy = -1, -1
        gx, gy = -1, -1
        kx, ky = -1, -1
        countA, countG, countK = 0, 0, 0
        ok = True
        for i in range(len(g)):
            for j in range(len(g[0])):
                if (i == 0 or i >= self.height - 1 or j == 0 or j >= self.width - 1) and g[i][j] != "w":
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

        dist = [[-1] * self.width for _ in range(self.height)]
        dist[sx][sy] = 0

        dir = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        while len(Q) > 0:
            x, y = Q.popleft()
            for dx, dy in dir:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.height and 0 <= ny < self.width):
                    continue
                if g[nx][ny] == "w":
                    continue
                if dist[nx][ny] == -1:
                    dist[nx][ny] = dist[x][y] + 1
                    Q.append([nx, ny])

        return dist[gx][gy] != -1 and dist[kx][ky] != -1

    def check_similarity(self, level1: str, level2: str):
        n = 0
        hit = 0
        for i in range(self.height):
            for j in range(self.width):
                c1 = level1[i][j]
                c2 = level2[i][j]
                if c1 == "\n":
                    continue
                if c1 == c2:
                    hit += 1
                n += 1
        return hit / n

    def get_property(self, level: str):
        index_agent, index_key, index_goal = 0, 0, 0
        for i, c in enumerate(level):
            if c == 'A':
                index_agent = i
            elif c == '+':
                index_key = i
            elif c == 'g':
                index_goal = i
        return (index_agent, index_key, index_goal)

    def evaluation(self, playable_levels: list[str]):
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
                if hamming / (self.height * self.width) <= 0.10:
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
