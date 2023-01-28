from __future__ import annotations
import numpy as np
from collections import deque
from .game.env import GameDescription, Game
from .config import BaseConfig
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def diversity_select(unique_playable_levels: list[str], config: BaseConfig, env: Game, feature_indices_dict: dict):
    def get_feature_count(level):
        feature = env.get_features(level)
        if feature in feature_indices_dict:
            return len(feature_indices_dict[feature])
        return 0

    unique_playable_levels = sorted(
        unique_playable_levels, key=get_feature_count)
    return unique_playable_levels[:2]


def kmeans_select(unique_playable_levels: list[str], config: BaseConfig, env: Game):
    def level_str_to_features(level_str):
        level_str = level_str.split()
        ret = np.zeros((len(level_str), len(level_str[0])))
        for i, s in enumerate(level_str):
            for j, c in enumerate(level_str[i]):
                ret[i, j] = env.ascii.index(c)
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
    n_clusters = elbow(levels_reduced)
    # print(f'n_clusters = {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(levels_reduced)
    indices = []

    # correct nearest centers
    for center in kmeans.cluster_centers_:
        dist = 100000
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
