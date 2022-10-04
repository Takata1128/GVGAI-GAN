import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from gan.env import Env
from gan.utils import check_playable_zelda


class SurrogateModelDataset(Dataset):
    def __init__(
        self,
        root,
        env: Env,
        datamode="train",
        transform=transforms.ToTensor(),
    ):
        self.env = env
        self.image_dir = os.path.join(root, env.name, datamode)
        self.image_paths = [
            os.path.join(self.image_dir, name) for name in os.listdir(self.image_dir)
        ]
        self.data_length = len(self.image_paths)

        self.transform = transform

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img, y = self._open(img_path)

        if not self.transform is None:
            img = self.transform(img)

        return img, y

    def _open(self, img_path):
        with open(img_path, "r") as f:
            level_str = f.read()
            level_str_list = f.readlines()
        ret = np.zeros(
            (len(self.env.ascii),
             self.env.state_shape[1], self.env.state_shape[2]),
        )
        for i, s in enumerate(level_str_list):
            for j, c in enumerate(s):
                if c == "\n":
                    break
                ret[self.env.ascii.index(c), i, j] = 1

        y = 1 if check_playable_zelda(level_str) else 0
        return ret, y
