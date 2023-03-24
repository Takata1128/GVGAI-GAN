from play_rl.config import Config
from play_rl.agent import Agent
from play_rl.env import Env


def main():
    config = Config()
    env_def = Env(
        "zelda"
    )
    agent = Agent(
        env_def,
        config
    )
    agent.train(
        num_env_steps=config.steps
    )


if __name__ == "__main__":
    main()
