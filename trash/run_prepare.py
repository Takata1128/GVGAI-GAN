from gan.prepare_dataset import prepare_dataset

if __name__ == "__main__":
    prepare_dataset(game_name='zelda')
    prepare_dataset(game_name='aliens')
    prepare_dataset(game_name='roguelike', version='v0')
