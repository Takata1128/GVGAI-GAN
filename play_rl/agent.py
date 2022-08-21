import os
import datetime
import time
import wandb
from collections import deque

import numpy as np
import torch
from .wrappers import make_vec_envs

from a2c_ppo_acktr import algo, utils
from .policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from .config import Config
from gan.env import Env


class Agent:
    def __init__(
        self,
        env_def: Env,
        config: Config,
    ):
        self.env_def = env_def
        self.config = config
        self.env_name = env_def.name
        self.algo = config.algo_name
        self.seed = config.seed
        self.num_processes = config.num_processes
        self.gamma = config.gamma
        self.model_save_dir = config.model_save_dir
        self.log_dir = config.log_dir
        self.cuda = config.cuda
        self.device = torch.device(
            "cpu") if not self.cuda else torch.device("cuda")
        self.num_steps = config.num_steps
        self.use_gae = config.use_gae
        self.gae_lambda = config.gae_lambda
        self.use_proper_time_limits = config.use_proper_time_limits
        self.recurrent_policy = config.recurrent_policy
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.lr = config.lr
        self.eps = config.eps
        self.max_grad_norm = config.max_grad_norm
        self.clip_param = config.clip_param
        self.ppo_epoch = config.ppo_epoch
        self.num_mini_batch = config.num_mini_batch
        self.alpha = config.alpha
        self.level_path = config.level_path
        self.project = config.project
        self.save_interval = config.save_interval
        self.log_interval = config.log_interval

        self.available_levels_ids = []
        self.non_available_level_ids = []

        self.envs = make_vec_envs(
            env_def=self.env_def,
            level_path=self.level_path,
            seed=self.seed,
            num_processes=self.num_processes,
            log_dir=self.log_dir,
            device=self.device,
            allow_early_resets=False,
        )

        self.actor_critic = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            base_kwargs={"recurrent": self.recurrent_policy},
        )
        self.actor_critic.to(self.device)

        self.agent = None
        if self.algo == "a2c":
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                self.value_loss_coef,
                self.entropy_coef,
                lr=self.lr,
                eps=self.eps,
                alpha=self.alpha,
                max_grad_norm=self.max_grad_norm,
            )
        elif self.algo == "ppo":
            self.agent = algo.PPO(
                self.actor_critic,
                self.clip_param,
                self.ppo_epoch,
                self.num_mini_batch,
                self.value_loss_coef,
                self.entropy_coef,
                lr=self.lr,
                eps=self.eps,
                max_grad_norm=self.max_grad_norm,
            )
        else:
            raise NotImplementedError("implemented only a2c or ppo.")

        if os.path.exists(
            os.path.join(self.model_save_dir, self.algo, self.env_name + ".pt")
        ):
            self.actor_critic = torch.load(
                os.path.join(self.model_save_dir, self.algo,
                             self.env_name + ".pt")
            )[0]

        self.rollouts = RolloutStorage(
            self.num_steps,
            self.num_processes,
            self.envs.observation_space.shape,
            self.envs.action_space,
            self.actor_critic.recurrent_hidden_state_size,
        )

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.cuda_deterministic = False
        if self.cuda and torch.cuda.is_available() and self.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        now = datetime.datetime.now()
        self.id = now.strftime("%Y%m%d%H%M%S")
        self.total_steps = 0

    def save(self, path, id):
        torch.save(
            self.actor_critic.state_dict(),
            os.path.join(path, "agent_{}.pth".format(id)),
        )

    """
    エージェントを学習
    """

    def train(
        self,
        num_env_steps,
        use_linear_lr_decay=False,
    ):
        if self.envs:
            self.envs.close()
        self.envs = make_vec_envs(
            self.env_def,
            self.level_path,
            self.seed,
            self.num_processes,
            self.log_dir,
            self.device,
            False,
        )

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        # エピソード報酬計算用
        episode_rewards = np.zeros(self.num_processes)
        episode_rewards_deque = deque(maxlen=30)

        start = time.time()
        num_updates = int(
            num_env_steps) // self.num_steps // self.num_processes

        wandb.login()
        metrics = {}
        with wandb.init(project=self.config.project, config=self.config.__dict__):
            for update_count in range(num_updates):
                if use_linear_lr_decay:
                    # decrease learning rate linearly
                    utils.update_linear_schedule(
                        self.agent.optimizer,
                        update_count,
                        num_updates,
                        self.lr,
                    )

                for step in range(self.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        (
                            value,
                            action,
                            action_log_prob,
                            recurrent_hidden_states,
                        ) = self.actor_critic.act(
                            self.rollouts.obs[step],
                            self.rollouts.recurrent_hidden_states[step],
                            self.rollouts.masks[step],
                        )

                    # Obser reward and next obs
                    obs, reward, done, infos = self.envs.step(action)

                    for i in range(self.num_processes):
                        episode_rewards[i] += reward[i]
                        if done[i]:
                            episode_rewards_deque.append(episode_rewards[i])
                            episode_rewards[i] = 0

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [
                            [0.0] if "bad_transition" in info.keys() else [1.0]
                            for info in infos
                        ]
                    )
                    self.rollouts.insert(
                        obs,
                        recurrent_hidden_states,
                        action,
                        action_log_prob,
                        value,
                        reward,
                        masks,
                        bad_masks,
                    )
                with torch.no_grad():
                    next_value = self.actor_critic.get_value(
                        self.rollouts.obs[-1],
                        self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1],
                    ).detach()
                self.rollouts.compute_returns(
                    next_value,
                    self.use_gae,
                    self.gamma,
                    self.gae_lambda,
                    self.use_proper_time_limits,
                )
                value_loss, action_loss, dist_entropy = self.agent.update(
                    self.rollouts)
                self.rollouts.after_update()

                # save for every interval-th episode or for the last epoch
                if (
                    update_count % self.save_interval == 0 or update_count == num_updates - 1
                ) and self.model_save_dir != "":
                    save_path = os.path.join(self.model_save_dir, self.algo)
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(
                        self.actor_critic.state_dict(),
                        os.path.join(save_path, self.env_name +
                                     "_" + self.id + ".pt"),
                    )

                if update_count % self.log_interval == 0 and len(episode_rewards_deque) > 1:
                    total_num_steps = (
                        (update_count + 1) * self.num_processes * self.num_steps
                    )
                    end = time.time()
                    print(
                        "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                            update_count,
                            total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards_deque),
                            np.mean(episode_rewards_deque),
                            np.median(episode_rewards_deque),
                            np.min(episode_rewards_deque),
                            np.max(episode_rewards_deque),
                        )
                    )
                    metrics["mean_rewards"] = np.mean(
                        episode_rewards_deque)
                    metrics["max_rewards"] = np.max(
                        episode_rewards_deque)
                    metrics["min_rewards"] = np.min(
                        episode_rewards_deque)
                    metrics["value_loss"] = value_loss
                    metrics["action_loss"] = action_loss
                    metrics["update_count"] = update_count
                    wandb.log(metrics, step=total_num_steps)
