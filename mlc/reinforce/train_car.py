import argparse
import re
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch

# from gymnasium.wrappers import RecordVideo
# from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from mlc.command.base import Base
from mlc.reinforce.car_nets import Modelo
from mlc.util.resources import get_time_as_str


class TrainCar(Base):

    def __init__(self, hparams):
        super().__init__(hparams)

        # try to use the device specified in the arguments
        self.device = "cpu"
        if hparams["device"].startswith("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device(hparams["device"])
            else:
                raise RuntimeError("CUDA is not available")
        self.hparams = hparams
        self.gamma = hparams["gamma"]

        self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/mlp_agent/{get_time_as_str()}"
        self.writer = SummaryWriter(self.output_folder + "/tensorboard")
        gym.register_envs(ale_py)

    @classmethod
    def name(cls):
        return "train_car"

    @staticmethod
    def add_arguments(parser):
        def _parse_device_arg(arg_value):
            pattern = re.compile(r"(cpu|cuda|cuda:\d+)")
            if not pattern.match(arg_value):
                raise argparse.ArgumentTypeError("invalid value")
            return arg_value

        parser.add_argument("-s", "--seed", type=int, default=42)  # TODO: use seed
        parser.add_argument("-e", "--max_episodes", type=int, default=100000)

        parser.add_argument("-g", "--game", default="CarRacing-v3")
        parser.add_argument("--num_envs", default=4, type=int)
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
        # parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=100, help="check point every n episodes")
        parser.add_argument("-v", "--video", type=int, default=100, help="create a video every n episodes")
        parser.add_argument("-p", "--personal", action="store_true", help="enable personal folder")
        parser.set_defaults(personal=False)
        parser.add_argument("-n", "--name", type=str, default=None, help="name this run")
        parser.add_argument("--gamma", type=float, default=0.9, help="discount factor for rewards")

    def run(self):
        # game = self.hparams["game"]
        torch.autograd.set_detect_anomaly(True)
        num_envs = self.hparams["num_envs"]
        # envs = gym.make_vec(game, render_mode=None, vectorization_mode="async", num_envs=num_envs)

        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False
                )
                for _ in range(num_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )

        device = "cpu"

        s, _ = envs.reset()
        s = torch.tensor(s, dtype=torch.float32).flatten(start_dim=1).to(device)

        n_actions = int(envs.action_space[0].n)
        all_actions = list(range(n_actions))
        policy_nn = Modelo(
            dim_input=s.shape[-1],
            dim_output=n_actions,
        ).to(device)

        learning_rate = torch.tensor(self.hparams["learning_rate"], dtype=torch.float32).to(device)
        optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)
        pbar = tqdm()

        states, info = envs.reset()
        states_new = torch.tensor(states, dtype=torch.float32).flatten(start_dim=1).to(device) / 255
        states_old = states_new
        states = states_new - states_old

        replay_buffers = []
        for i in range(envs.num_envs):
            replay_buffers.append(deque(maxlen=4096))

        episode_start = np.zeros(envs.num_envs, dtype=bool)

        # max_reward = -9999
        n_episodes = 0
        while True:

            with torch.no_grad():
                action_dist = policy_nn(states).cpu().detach().numpy()

            # sample actions
            actions = []
            for i in range(num_envs):
                actions.append(np.random.choice(all_actions, p=action_dist[i]))

            # vectorized step
            aux_states, rewards, terminations, truncations, info = envs.step(actions)

            states_old = states_new
            states_new = torch.tensor(aux_states, dtype=torch.float32).flatten(start_dim=1).to(device) / 255

            for i in range(envs.num_envs):

                if not episode_start[i]:  # se n acabou no ciclo passado
                    replay_buffers[i].append(
                        {
                            "state": states[i],
                            "frame": aux_states[i],
                            "action": int(actions[i]),
                            "reward": float(rewards[i]),
                            "termination": bool(terminations[i]),
                            "truncation": bool(truncations[i]),
                        }
                    )

            states = states_new - states_old
            episode_start = np.logical_or(terminations, truncations)

            for i in range(envs.num_envs):
                if episode_start[i]:  # se acabou o episódio nessa iteração
                    n_episodes += 1
                    pbar.update()
                    replay = list(replay_buffers[i])  # replay do agente i

                    # find start of last termination
                    j0 = -1
                    for j, r in enumerate(replay[:-1]):
                        if r["termination"] or r["truncation"]:
                            j0 = j
                    j0 += 1
                    replay = replay[j0:]

                    rewards = [x["reward"] for x in replay]
                    sum_rewards = sum(rewards)
                    self.writer.add_scalar("reward", sum_rewards, n_episodes)

                    propagated_rewards = []

                    running_mean = 0
                    for R in rewards[::-1]:

                        running_mean = R + self.gamma * running_mean
                        propagated_rewards.insert(0, running_mean)

                    n_nonzero_rewards = sum([abs(x) > 0.5 for x in rewards])
                    replay_states = torch.stack([x["state"] for x in replay])

                    if n_nonzero_rewards >= 0:
                        preds = policy_nn(replay_states)
                        loss = torch.tensor(0, dtype=torch.float32).to(device)
                        for j, r in enumerate(replay):
                            loss += (
                                -torch.log(10e-9 + preds[j][r["action"]]) * propagated_rewards[j] / n_nonzero_rewards
                            )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if (n_episodes - 1) % self.hparams["video"] == 0:
                        frames = np.stack([x["frame"] for x in replay])
                        frames = np.permute_dims(frames, (0, 3, 1, 2))
                        frames = np.expand_dims(frames, axis=0)
                        self.writer.add_video("gameplay", frames, n_episodes, fps=30)

                    self.writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainCar.add_arguments(parser)

    args = parser.parse_args()
    hparams = vars(args)
    t = TrainCar(hparams)
    t.run()
