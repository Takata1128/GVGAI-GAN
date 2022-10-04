from gan.utils import make_zelda_level, check_playable_zelda
import os

if __name__ == '__main__':
    dir_path = os.path.dirname(__file__) + f"/gan/data/level/zelda/random/"
    for i in range(100):
        str = make_zelda_level()
        if check_playable_zelda(str):
            with open(
                dir_path + f"zelda_{i}", mode="w"
            ) as f:
                f.write(str)
