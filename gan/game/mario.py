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
