from __future__ import annotations
from .env import Game
from collections import deque
import wandb


class Boulderdash(Game):
    def __init__(self):
        super().__init__('boulderdash', 'v0')
        self.height = self.map_shape[0]
        self.width = self.map_shape[1]
        self.padding_index = 0

    def get_features(self, level_str: str):
        agent_index = -1
        goal_index = -1
        enemy_num, block_num, diamond_num = 0, 0, 0
        for i, c in enumerate(level_str):
            if c == 'A':
                agent_index = i
            if c == 'e':
                goal_index = i
            if c in ['b', 'c']:
                enemy_num += 1
            if c == '.':
                block_num += 1
            if c == 'x':
                diamond_num += 1

        return (agent_index, goal_index, enemy_num // 3, diamond_num // 5)

    def evaluation(self, playable_levels: list[str]):
        def check_level_hamming(level1: str, level2: str):
            hit = 0
            for c1, c2 in zip(level1, level2):
                if c1 == "\n":
                    continue
                if c1 != c2:
                    hit += 1
            return hit

        levels_small_set = playable_levels[:1000]
        total_hamming_dist, features_duplication, n = 0, 0, 0
        similarity_threshold_list = [0.60, 0.65,
                                     0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        duplication_rate_list = [0 for i in range(
            len(similarity_threshold_list))]

        features_dict = {}

        for i in range(len(levels_small_set)):
            i_features = self.get_features(levels_small_set[i])
            if i_features not in features_dict:
                features_dict[i_features] = 1
            else:
                features_dict[i_features] += 1

            for j in range(len(levels_small_set)):
                if i <= j:
                    continue
                j_features = self.get_features(levels_small_set[j])
                features_duplication += 1 if (i_features == j_features) else 0
                hamming = check_level_hamming(
                    levels_small_set[i], levels_small_set[j])
                total_hamming_dist += hamming
                for k in range(len(similarity_threshold_list)):
                    if hamming / (self.height * self.width) <= (1 - similarity_threshold_list[k]):
                        duplication_rate_list[k] += 1
                n += 1

        for i in range(len(duplication_rate_list)):
            duplication_rate_list[i] /= n

        unique_levels = list(set(playable_levels))

        metrics = {}
        metrics['wandb'] = {}
        metrics['other'] = {}
        metrics['wandb']["Final Duplication Rate"] = 1 - \
            len(unique_levels) / len(playable_levels)
        print("Duplication Rate:", 1 - len(unique_levels) / len(playable_levels))
        metrics['wandb']["Hamming Distance"] = total_hamming_dist / n
        print("Hamming Distance:", total_hamming_dist / n)
        # data = [[x, y] for (x, y) in zip(
        #     similarity_threshold_list, duplication_rate_list)]
        # table = wandb.Table(data=data, columns=['rate', 'duplications'])
        # wandb_metrics[r'X% Duplication Rate'] = wandb.plot.line(
        #     table, 'rate', 'duplications')
        # wandb_metrics[r'X% Duplication Rate']
        metrics['wandb']["Features Duplication Rate"] = features_duplication / n
        metrics['wandb']["Features Type Nums"] = len(features_dict)
        metrics['other'][r'X% Duplication Rate'] = (
            similarity_threshold_list, duplication_rate_list)

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
        level1 = level1.split()
        level2 = level2.split()
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
