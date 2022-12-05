from __future__ import annotations
from .env import Game
from collections import deque


class Mario(Game):
    def __init__(self, name, version):
        super().__init__(name, version)
        self.height = 14
        self.width = 28

    def check_playable(self, lvl_str: str):
        g = lvl_str.split()
        ok = True
        for i, row in enumerate(g[:self.height]):
            for j, c in enumerate(row[:self.width]):
                if c == '<':
                    if j != self.width - 1 and g[i][j + 1] != '>':
                        ok = False
                    if i != 0 and (g[i + 1][j] != '['):
                        ok = False
                if c == '>':
                    if j != 0 and g[i][j - 1] != '<':
                        ok = False
                    if i != 0 and (g[i + 1][j] != ']'):
                        ok = False
                if c == '[':
                    if j != self.width - 1 and g[i][j + 1] != ']':
                        ok = False
                    if i != self.height - 1 and (g[i + 1][j] not in ['[', 'X']):
                        ok = False
                    if i != 0 and (g[i - 1][j] not in ['[', '<']):
                        ok = False
                if c == ']':
                    if j != 0 and g[i][j - 1] != '[':
                        ok = False
                    if i != self.height - 1 and (g[i + 1][j] not in [']', 'X']):
                        ok = False
                    if i != 0 and (g[i - 1][j] not in [']', '>']):
                        ok = False
        if not ok:
            return False

        if not self.check_reachable(g):
            ok = False

        return ok

    def check_reachable(self, g: list[str]):
        visited = [[0 for i in range(self.width)] for j in range(self.height)]
        que = deque()
        ok = False
        # スタート地点　横幅4マス分をチェック　最も低い位置をスタート地点
        for x in range(0, 4):
            for y in range(self.height - 2, 0, -1):
                if g[y][0] == '-' and g[y + 1][0] in ['X', 'S', 'Q']:
                    visited[y][0] = 1
                    que.append((y, 0))
                    break
        while que:
            current_y, current_x = que.popleft()
            for target_x in range(max(current_x - 4, 0), min(current_x + 4 + 1, self.width)):  # 横は4マスまで
                # 上方向(座標小さくなる方向)4マスまで 下方向いくらでも
                for target_y in range(max(current_y - 4, 0), self.height - 1):
                    if (current_y, current_x) == (target_y, target_x):
                        continue
                    if g[target_y][target_x] in ['-', 'E'] and g[target_y + 1][target_x] in ['X', 'S', 'Q']:  # 上に乗れるマス
                        if visited[target_y][target_x] == 1:
                            continue
                        ### 障害物はない？ ###
                        is_obs = False
                        yy, xx = current_y, current_x
                        if target_y < current_y:  # 高い
                            # まずは上方向
                            while yy != target_y:
                                if g[yy][current_x] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                    is_obs = True
                                yy -= 1
                            # 横方向
                            if target_x < current_x:  # 手前
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                        is_obs = True
                                    xx -= 1
                            else:  # 奥
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                        is_obs = True
                                    xx += 1
                        elif target_y == current_y:  # 同じ
                            # 横方向
                            if target_x < current_x:  # 手前
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                        is_obs = True
                                    xx -= 1
                            else:  # 奥
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                        is_obs = True
                                    xx += 1
                        else:  # 低い
                            # まずは横方向
                            if target_x < current_x:  # 手前
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                        is_obs = True
                                    xx -= 1
                            else:  # 奥
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                        is_obs = True
                                    xx += 1
                            # 下方向
                            while yy != target_y:
                                if g[yy][current_x] in ['X', 'S', 'Q', '<', '>', '[', ']']:
                                    is_obs = True
                                yy += 1
                        if is_obs:
                            continue
                        visited[target_y][target_x] = 1
                        que.append((target_y, target_x))
        for target_x in range(self.height):  # 最終列に到達可能か
            if visited[target_x][self.width - 1] == 1:
                ok = True
        return ok

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

        levels_small_set = playable_levels[:1000]
        total_hamming_dist, dup90, n = 0, 0, 0
        for i in range(len(levels_small_set)):
            for j in range(len(levels_small_set)):
                if i < j:
                    continue
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
        print("Duplication Ratio:", 1 - len(unique_levels) / len(playable_levels))
        print("Hamming Distance:", total_hamming_dist / n)
        return metrics
