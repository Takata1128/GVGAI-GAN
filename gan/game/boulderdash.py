from __future__ import annotations
from .env import Game
from collections import deque


class Boulderdash(Game):
    def __init__(self, name: str, version: str):
        super().__init__(name, version)
        self.height = self.map_shape[0]
        self.width = self.map_shape[1]
        self.padding_index = 0

    def get_features(self, level_str: str):
        agent_index = -1
        goal_index = -1
        enemy_num = 0
        for i, c in enumerate(level_str):
            if c == 'A':
                agent_index = i
            if c == 'e':
                goal_index = i
            if c in ['b', 'c']:
                enemy_num += 1

        return (agent_index, goal_index, enemy_num // 3)

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
            g, a, all = 0, 0, 0
            for c1, c2 in zip(level1, level2):
                if c1 == "\n":
                    continue
                if c1 == 'e' and c1 == c2:
                    g = 1
                if c1 == 'A' and c1 == c2:
                    a = 1
            if g and a:
                all = 1
            return g, a, all

        key_duplication, goal_duplication, player_duplication, total_object_duplication, total_hamming_dist, dup90, n = 0, 0, 0, 0, 0, 0, 0
        levels_small_set = playable_levels[:1000]
        for i in range(len(levels_small_set)):
            for j in range(len(levels_small_set)):
                if i < j:
                    continue
                gd, pd, sod = check_level_object_duprecated(
                    levels_small_set[i], levels_small_set[j])
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
        metrics["Final Duplication Rate"] = 1 - \
            len(unique_levels) / len(playable_levels)
        metrics["Hamming Distance"] = total_hamming_dist / n
        metrics[r"90% duplication Rate"] = dup90 / n
        metrics["Object Duplication Rate"] = total_object_duplication / n
        metrics["Key Duplication Rate"] = key_duplication / n
        metrics["Goal Duplication Rate"] = goal_duplication / n
        metrics["Player Duplication Rate"] = player_duplication / n
        print("Duplication Rate:", 1 - len(unique_levels) / len(playable_levels))
        print("Hamming Distance:", total_hamming_dist / n)
        print("Obj Duplication Rate:", total_object_duplication / n)

        return metrics

    def check_playable(self, lvl_str: str):
        g = lvl_str.split()
        sx, sy = -1, -1
        gx, gy = -1, -1
        diamond_coods = []
        countA, countG = 0, 0
        ok = True
        for i in range(self.height):
            for j in range(self.width):
                # まわりは壁
                if (i == 0 or i >= self.height - 1 or j == 0 or j >= self.width - 1) and g[i][j] != "w":
                    ok = False

                # 必要なオブジェクト
                if g[i][j] == "A":
                    sx, sy = i, j
                    countA += 1
                if g[i][j] == "e":
                    gx, gy = i, j
                    countG += 1
                if g[i][j] == "x":
                    diamond_coods.append((i, j))

        if not (countA == 1 and countG == 1 and len(diamond_coods) >= 10) or not ok:
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
                if g[nx][ny] in ["w", 'o']:
                    continue
                if dist[nx][ny] == -1:
                    dist[nx][ny] = dist[x][y] + 1
                    Q.append([nx, ny])

        ok = (dist[gx][gy] != -1)
        for dia_x, dia_y in diamond_coods:
            if dist[dia_x][dia_y] == -1:
                ok = False

        return ok

    def check_similarity(self, level1: str, level2: str):
        n = 0
        hit = 0
        for i in range(self.height):
            for j in range(self.width):
                c1 = level1[i][j]
                c2 = level2[i][j]
                if c1 == c2:
                    hit += 1
                n += 1
        return hit / n
