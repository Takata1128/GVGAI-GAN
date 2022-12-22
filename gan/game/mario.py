from __future__ import annotations
from .env import Game
from collections import deque
import wandb

COVER_THRESHOLD = 0.20
TILE_TYPE_THRESHOLD = 3


class Mario(Game):
    def __init__(self):
        super().__init__('mario', 'v0')
        self.height = 14
        self.width = 28
        self.padding_index = 0

    def check_playable(self, lvl_str: str):
        g = lvl_str.split()
        if not self._check_pipe(g):
            return False

        if not self._check_teritory(g):
            return False

        if not self._check_tile_balance(g):
            return False

        ok, _ = self._check_reachable(g)
        if not ok:
            return False
        return ok

    def _check_teritory(self, g: list[str]):
        '''
        領域が複数あるかどうかチェック
        '''
        filled = [[0 for _ in range(self.width)] for _ in range(self.height)]
        sx, sy = -1, -1
        for i, row in enumerate(g[:self.height]):
            for j, c in enumerate(row[:self.width]):
                if c == '-':
                    sx, sy = i, j
                    break
            if sx != -1:
                break

        if sx == -1:
            return False

        que = deque()
        filled[sx][sy] = 1
        que.append((sx, sy))
        dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while que:
            current_x, current_y = que.popleft()
            for dx, dy in dir:
                nx, ny = current_x + dx, current_y + dy
                if not (0 <= nx < self.height and 0 <= ny < self.width) or (g[nx][ny] not in ['-', 'E']):
                    continue
                if filled[nx][ny] == 0:
                    filled[nx][ny] = 1
                    que.append((nx, ny))

        ok = True
        for i in range(self.height):
            for j in range(self.width):
                if g[i][j] == '-' and filled[i][j] == 0:
                    ok = False
        return ok

    def _check_pipe(self, g: list[str]):
        '''
        土管が壊れていないかチェック        
        '''
        ok = True
        for i, row in enumerate(g[:self.height]):
            for j, c in enumerate(row[:self.width]):
                if c == '<':
                    if j != self.width - 1 and g[i][j + 1] != '>':
                        ok = False
                    if i != self.height - 1 and (g[i + 1][j] != '['):
                        ok = False
                if c == '>':
                    if j != 0 and g[i][j - 1] != '<':
                        ok = False
                    if i != self.height - 1 and (g[i + 1][j] != ']'):
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
        return ok

    def _check_tile_balance(self, g: list[str]):
        '''
        タイルの種類数と全体に占める割合をチェック
        上3列見てブロックがあればNG
        '''
        tile_dict = {}
        ok = True
        for i, row in enumerate(g[:self.height]):
            for j, c in enumerate(row[:self.width]):
                if i < 3 and c != '-':
                    return False
                if c in tile_dict:
                    tile_dict[c] += 1
                else:
                    tile_dict[c] = 1
        for c, num in tile_dict.items():
            if c == '-':
                continue
            if num >= COVER_THRESHOLD * (self.height * self.width):
                ok = False

        if len(tile_dict) < TILE_TYPE_THRESHOLD:
            ok = False
        return ok

    def _check_reachable(self, g: list[str]):
        '''
        ゴールに到達できるかどうか
        '''
        visited = [[0 for i in range(self.width)] for j in range(self.height)]
        que = deque()
        ok = False
        # スタート地点　横幅4マス分をチェック　最も低い位置をスタート地点
        for x in range(0, 4):
            for y in range(self.height - 2, 0, -1):
                if g[y][x] in ['-', 'E'] and g[y + 1][x] in ['X', 'S', 'Q', '<', '>', '?']:
                    visited[y][x] = 1
                    que.append((y, x))
                    break
        while que:
            current_y, current_x = que.popleft()
            for target_x in range(max(current_x - 4, 0), min(current_x + 4 + 1, self.width)):  # 横は4マスまで
                # 上方向(座標小さくなる方向)4マスまで 下方向いくらでも
                for target_y in range(max(current_y - 4, 0), self.height - 1):
                    if (current_y, current_x) == (target_y, target_x):
                        continue
                    if g[target_y][target_x] in ['-', 'E'] and g[target_y + 1][target_x] in ['X', 'S', 'Q', '?', '<', '>']:  # 上に乗れるマス
                        if visited[target_y][target_x] == 1:
                            continue
                        ### 障害物はない？ ###
                        is_obs = False
                        yy, xx = current_y, current_x
                        if target_y < current_y:  # 高い
                            # まずは上方向
                            while yy != target_y:
                                if g[yy][current_x] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                    is_obs = True
                                yy -= 1
                            # 横方向
                            if target_x < current_x:  # 手前
                                while xx != target_x:
                                    if g[yy][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                        is_obs = True
                                    xx -= 1
                            else:  # 奥
                                while xx != target_x:
                                    if g[yy][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                        is_obs = True
                                    xx += 1
                        elif target_y == current_y:  # 同じ
                            # 横方向
                            if target_x < current_x:  # 手前
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                        is_obs = True
                                    xx -= 1
                            else:  # 奥
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                        is_obs = True
                                    xx += 1
                        else:  # 低い
                            # まずは横方向
                            if target_x < current_x:  # 手前
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                        is_obs = True
                                    xx -= 1
                            else:  # 奥
                                while xx != target_x:
                                    if g[current_y][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                        is_obs = True
                                    xx += 1
                            # 下方向
                            while yy != target_y:
                                if g[yy][xx] in ['X', 'S', 'Q', '<', '>', '[', ']', '?']:
                                    is_obs = True
                                yy += 1
                        if is_obs:
                            continue
                        visited[target_y][target_x] = 1
                        que.append((target_y, target_x))

        # 後ろの方に到達可能か
        reachable = False
        for x in range(self.width - 4, self.width):
            for y in range(self.height):
                if visited[y][x] == 1:
                    reachable = True

        if reachable:
            ok = True

        return ok, visited

    def check_similarity(self, level1: str, level2: str):
        level1 = level1.split()
        level2 = level2.split()
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

    def get_features(self, level: str):
        g = level.split()
        num_pipe, num_hole = 0, 0
        for i in range(self.height):
            for j in range(self.width):
                if i == self.height - 1 and g[i][j] == '-':
                    num_hole += 1
                if g[i][j] == '<':
                    num_pipe += 1
        return (num_pipe, num_hole // 3)

    def evaluation(self, playable_levels: list[str]):
        def check_level_hamming(level1: str, level2: str):
            l1 = level1.split()
            l2 = level2.split()
            hit = 0
            for i in range(self.height):
                for j in range(self.width):
                    c1, c2 = l1[i][j], l2[i][j]
                    if c1 != c2:
                        hit += 1
            return hit

        levels_small_set = playable_levels[:1000]
        total_hamming_dist, n = 0, 0
        similarity_threshold_list = [0.60, 0.65,
                                     0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        duplication_rate_list = [0 for i in range(
            len(similarity_threshold_list))]

        for i in range(len(levels_small_set)):
            for j in range(len(levels_small_set)):
                if i <= j:
                    continue
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
        metrics["Final Duplication Rate"] = 1 - \
            len(unique_levels) / len(playable_levels)
        metrics["Hamming Distance"] = total_hamming_dist / n
        data = [[x, y] for (x, y) in zip(
            similarity_threshold_list, duplication_rate_list)]
        table = wandb.Table(data=data, columns=['rate', 'duplications'])
        metrics[r'X% Duplication Rate'] = wandb.plot.line(
            table, 'rate', 'duplications')
        print("Duplication Rate:", 1 - len(unique_levels) / len(playable_levels))
        print("Hamming Distance:", total_hamming_dist / n)
        return metrics
