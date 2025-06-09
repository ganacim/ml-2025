import argparse
import gymnasium as gym
from time import sleep
import random
import torch
from torch import nn
from mlc.reinforcement.network import MLP
import numpy as np
from tqdm import tqdm

class ExperienceReplay:
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.experience = [None]*capacity
        self.n = 0

    def append(self, s0, a0, r0, s1):
        self.experience[self.n % self.capacity] = [s0, a0, r0, s1]
        self.n += 1

    def sample(self, k=128):
        if self.n < self.capacity:
            return random.sample(self.experience[:self.n], k=self.n)
        return random.sample(self.experience, k=k)




def main():
    torch.autograd.set_detect_anomaly(True)
    env = gym.make("LunarLander-v3", render_mode=None)

    device = 'cpu'

    gamma = .99 # reward decay
    replay_capacity = 4096

    experience_replay = ExperienceReplay(capacity=replay_capacity)

    policy_nn = MLP(
        dim_input = env.observation_space.shape[0],
        dim_output = int(env.action_space.n)
    ).to(device)

    loss_fn = nn.MSELoss().to(device)
    episodes = 0
    learning_rate = .01
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)
    best_reward = -9999999999
    sum_reward = 0
    pbar = tqdm()
    while 1:
        n_eval = 100
        if episodes % n_eval == 0:
            env = gym.make("LunarLander-v3", render_mode='human')

        s1, info = env.reset()
        s1 = torch.tensor(s1, dtype=torch.float32).to(device)
        episode_over = False

        while not episode_over:
            s0 = s1.clone()
            with torch.no_grad():
                a0 = policy_nn(s0)

            if random.random() < .05:
                action = env.action_space.sample()
            else:
                action = np.argmax(a0.cpu().detach().numpy())
            s1, r0, terminated, truncated, info = env.step(action)

            s1 = torch.tensor(s1, dtype=torch.float32).to(device)
            r0 = torch.tensor(r0, dtype=torch.float32).to(device)
            experience_replay.append(s0, a0, r0, s1)
            episode_over = terminated or truncated
            pbar.update()

            #for s0, a0, r0, s1 in experience_replay.sample(128):
            with torch.no_grad():
                if episode_over:
                    y = r0
                else:
                    y = r0 + gamma*torch.max(policy_nn(s1))

            optimizer.zero_grad()
            loss = loss_fn(y, torch.max(policy_nn(s0)))
            loss.backward()
            optimizer.step()

        best_reward = max(float(r0), best_reward)
        sum_reward += float(r0)

        if episodes % n_eval == 0:
            print()
            print(f"{best_reward=}")
            print(f"{sum_reward/n_eval=}")
            sum_reward = 0
            env = gym.make("LunarLander-v3", render_mode=None)

        episodes += 1

    env.close()

if __name__ == "__main__":
    main()
