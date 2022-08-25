import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from gan.env import Env


class LevelDataset(Dataset):
    def __init__(
        self,
        root,
        env: Env,
        datamode="train",
        transform=transforms.ToTensor(),
        latent_size=100,
    ):
        self.env = env
        self.image_dir = os.path.join(root, env.name, datamode)
        self.image_paths = [
            os.path.join(self.image_dir, name) for name in os.listdir(self.image_dir)
        ]
        self.data_length = len(self.image_paths)

        self.transform = transform
        self.latent_size = latent_size

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        latent = torch.randn(size=(self.latent_size,))
        img_path = self.image_paths[index]
        img, label = self._open(img_path)

        if not self.transform is None:
            img = self.transform(img)

        return latent, img, label

    def _open(self, img_path):
        with open(img_path, "r") as f:
            datalist = f.readlines()
        ret = np.zeros(
            (len(self.env.ascii),
             self.env.state_shape[1], self.env.state_shape[2]),
        )

        # label : onehot vector of counts of map tile object.
        label = np.zeros(len(self.env.ascii))
        for i, s in enumerate(datalist):
            for j, c in enumerate(s):
                if c == "\n":
                    break
                ret[self.env.ascii.index(c), i, j] = 1
                label[self.env.ascii.index(c)] += 1
        return ret, label
