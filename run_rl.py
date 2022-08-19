from play_rl.config import Config
from play_rl.agent import Agent
from gan.env import Env


def main():
    config = Config()
    env_def = Env(
        "zelda",
        version='v1'
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
