import torch
import cma
import numpy as np
import os
from gan.small_models import Generator
from gan.config import SmallModelConfig
from gan.utils import tensor_to_level_str, check_playable


def optimize(config: SmallModelConfig):
    if config.cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available else "cpu")
        print("device : cuda")
    else:
        device = torch.device("cpu")
        print("device : cpu")
    generator = Generator(
        out_dim=config.input_shape[0],
        shapes=config.model_shapes,
        z_shape=(config.latent_size,),
        filters=config.generator_filters,
        use_conditional=config.use_conditional
    ).to(device)

    model_path = os.path.join(
        "/root/mnt/pcg/GVGAI-GAN/gan/checkpoints/none-795", "models.tar")
    load_model = torch.load(model_path)
    generator.load_state_dict(load_model["generator"])

    x = torch.randn(config.latent_size).to(device)

    # es = cma.CMAEvolutionStrategy(config.latent_size*[0], 0.5)
    # es.optimize(fitness)

    def fitness(x: torch.Tensor):
        x = np.array(x)
        latent = torch.FloatTensor(x).view(1, -1, 1, 1)
        level = generator(x)
        level_str = tensor_to_level_str(level)
        playable = check_playable(level_str)
        eval = eval(level_str)
        return playable*300 + eval

    def eval(level_str):
        wall = 0
        enemy = 0
        for s, i in enumerate(level_str):
            for c, j in enumerate(level_str):
                if c == 'w':
                    wall += 1
                if c in ['1', '2', '3']:
                    enemy += 1
        return wall + enemy


if __name__ == '__main__':
    pass
