import argparse
import re
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import random
# from gymnasium.wrappers import RecordVideo
# from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F


from mlc.command.base import Base
from mlc.reinforcement.networks import MLP, CNN
from mlc.util.resources import get_time_as_str


class Train(Base):

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
        
        self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/mlp_agent/{get_time_as_str()}"
        self.writer = SummaryWriter(self.output_folder + "/tensorboard")
        self.action_names = hparams["action_names"]
        self.image_size = hparams["image_size"]
        self.nframes = hparams["nframes"]
        gym.register_envs(ale_py)
    @classmethod
    def name(cls):
        return "mlp_agent.train"

    @staticmethod
    def add_arguments(parser):
        def _parse_device_arg(arg_value):
            pattern = re.compile(r"(cpu|cuda|cuda:\d+)")
            if not pattern.match(arg_value):
                raise argparse.ArgumentTypeError("invalid value")
            return arg_value

        parser.add_argument("-s", "--seed", type=int, default=42)  # TODO: use seed
        parser.add_argument("-ep", "--epsilon", type=float, default=0.05)
        parser.add_argument("-ew", "--entropy-weight", type=float, default=0.00035)
        parser.add_argument("-e", "--max_episodes", type=int, default=100000)
        parser.add_argument("-ne", "--num_episodes", type=int, default=3)
        parser.add_argument("-nen", "--num_episodes_naive", type=int, default=25)
        parser.add_argument("-nc", "--naive_chance_up", type=int, default=0.75)
        parser.add_argument("-g", "--game", default="ALE/Freeway-v5")
        parser.add_argument("-nf", "--nframes", type=int, default=4, help = "Number of game frames to use")
        parser.add_argument("-is", "--image-size",type=int, default=84, help="Image size to use")
        parser.add_argument("--num_envs", default=4, type=int)
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
        # parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=100, help="check point every n episodes")
        parser.add_argument("-v", "--video", type=int, default=100, help="create a video every n episodes")
        parser.add_argument("-p", "--personal", action="store_true", help="enable personal folder")
        parser.set_defaults(personal=False)
        parser.add_argument("-n", "--name", type=str, default=None, help="name this run")
        parser.add_argument("-an", "--action-names", type=str, default=["NOOP","UP", "DOWN"], help="Name actions")

    def crop_env(self, frame):
        return frame[10:10+84,:,:]

    def make_env(self, game):
        env = gym.make(game, difficulty = 1)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (self.image_size,self.image_size))
        env = gym.wrappers.FrameStackObservation(env, self.nframes)
        return env
    
    def run(self):
        game = self.hparams["game"]
        torch.autograd.set_detect_anomaly(True)
        num_envs = self.hparams["num_envs"]
        epsilon = self.hparams["epsilon"]
        entropy_weight = self.hparams["entropy_weight"]
        past_ep_to_consider = hparams["num_episodes"]
        episodes_w_naivestrat = hparams["num_episodes_naive"]
        naive_chance_up = hparams["naive_chance_up"]
        # envs = gym.make_vec(game, render_mode=None, vectorization_mode="async", num_envs=num_envs)
         
        envs = gym.vector.AsyncVectorEnv(
            [lambda: self.make_env(game) for _ in range(num_envs)], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )

        device = "cpu"
        n_actions = int(envs.action_space[0].n)
        all_actions = list(range(n_actions))
        policy_nn = CNN(
            nframes=self.nframes,
            im_sz=self.image_size,
            nactions=n_actions
        ).to(device)
        learning_rate = torch.tensor(self.hparams["learning_rate"], dtype=torch.float32).to(device)
        optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)
        pbar = tqdm()
        
        states, info = envs.reset()
        states = torch.tensor(states, dtype=torch.float32).permute(0, 1, 3, 2).to(device) / 255

        replay_buffers = []
        for i in range(envs.num_envs):
            replay_buffers.append(deque(maxlen=4096))

        episode_start = np.zeros(envs.num_envs, dtype=bool)

        # max_reward = -9999
        n_episodes = 0
        while True:
            with torch.no_grad():
                action_dist = torch.softmax(policy_nn(states), dim=1).cpu().detach().numpy()
            # sample actions
            actions = []
            for i in range(num_envs):
                sample = random.random()
                if n_episodes < episodes_w_naivestrat:
                    if random.random() < naive_chance_up:
                        actions.append(1) #UP
                    else:
                        actions.append(random.choice(list(range(n_actions))))
                elif sample < epsilon:
                    actions.append(random.choice(list(range(n_actions))))
                else:
                    actions.append(np.random.choice(all_actions, p=action_dist[i]))

            # vectorized step
            aux_states, rewards, terminations, truncations, info = envs.step(actions)

            states_new = torch.tensor(aux_states, dtype=torch.float32).permute(0, 1, 3, 2).to(device) / 255
            for i in range(envs.num_envs):

                if not episode_start[i]:
                    if rewards[i] == 0:
                            rewards[i] = -0.001
                    replay_buffers[i].append(
                        {
                            "state": states[i],
                            "frame": aux_states[i],
                            "policy": action_dist[i],
                            "action": int(actions[i]),
                            "reward": float(rewards[i]),
                            "termination": bool(terminations[i]),
                            "truncation": bool(truncations[i]),
                        }
                    )

            states = states_new
            episode_start = np.logical_or(terminations, truncations)

            for i in range(envs.num_envs):
                if episode_start[i]:
                    n_episodes += 1
                    pbar.update()
                    replay = list(replay_buffers[i])

                    # find start of last termination
                    j0 = -1
                    j_list = [0]
                    for j, r in enumerate(replay[:]):
                        if r["termination"] or r["truncation"]:
                            j_list.append(j)
                    n_keep = min(len(j_list), past_ep_to_consider)
                    j0 = j_list[-(n_keep)]
                    replay = replay[j0:]
                    rewards = [x["reward"] for x in replay]
                    sum_rewards = sum(rewards) / n_keep
                    
                    self.writer.add_scalar("reward", sum_rewards, n_episodes)
                    
                    
                    propagated_rewards = []

                    running_mean = 0
                    for R in rewards[::-1]:
                        #if abs(R) > 0.5:
                        #    running_mean = 0
                        running_mean = R + 0.99 * running_mean
                        propagated_rewards.insert(0, running_mean)
                    propagated_rewards = torch.tensor(propagated_rewards)
                                                      
                    #n_nonzero_rewards = sum([abs(x) > 0.5 for x in rewards])
                    replay_states = torch.stack([x["state"] for x in replay])
                    replay_actions = torch.tensor([x["action"] for x in replay], dtype=torch.int64).to(device)
                    #propagated_rewards = (propagated_rewards - propagated_rewards.mean()) / (propagated_rewards.std() + 1e-9)


                    preds = policy_nn(replay_states)
                    loss = torch.tensor(0, dtype=torch.float32).to(device)
                    #every 25 episodes, log action probabilities for debug
                    if n_episodes % 25 == 0:
                        probs = F.softmax(preds, dim=1)
                        self.writer.add_histogram("policy_probs", probs, n_episodes)
                        for i in range(preds.shape[-1]):
                            self.writer.add_scalar(f"policy_probs/mean_action_{self.action_names[i]}", probs[:, i].mean().item(), n_episodes)
                            self.writer.add_scalar(f"policy_probs/std_action_{self.action_names[i]}", probs[:, i].std().item(), n_episodes)
                    log_preds = F.cross_entropy(preds, replay_actions, reduction = "none")
                    #selected_log_probs = log_preds[range(len(replay_actions)), replay_actions]
                    policy_score = torch.sum(log_preds * propagated_rewards)
                    #entropy = -torch.mean(torch.sum(preds * log_preds))
                    #loss = policy_loss - entropy_weight * entropy
                    loss = policy_score 

                    #self.writer.add_scalar("entropy", (entropy_weight*entropy).item(), n_episodes)
                    self.writer.add_scalar("policy score", policy_score.item(), n_episodes)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (n_episodes - 1) % self.hparams["video"] == 0:
                        frames = np.stack([x["frame"] for x in replay])[:,0,:,:].reshape(-1,1,84,84)
                        frames = np.repeat(frames, 3, axis=1)
                        frames = np.permute_dims(frames, (0, 1, 3, 2))
                        frames = np.rot90(frames, k=1, axes=(3, 2))
                        frames = np.flip(frames, axis=3)
                        frames = np.expand_dims(frames, axis=0)
                        self.writer.add_video("gameplay", frames, n_episodes, fps=30)

                    self.writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Train.add_arguments(parser)

    args = parser.parse_args()
    hparams = vars(args)
    t = Train(hparams)
    t.run()
