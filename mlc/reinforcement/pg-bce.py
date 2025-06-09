import argparse
import gymnasium as gym
from time import sleep
import random
import torch
from torch import nn
from mlc.reinforcement.network import MLPBCE
import numpy as np
from tqdm import tqdm
import ale_py


gym.register_envs(ale_py)

def main():
    game ="ALE/Pong-v5"
    torch.autograd.set_detect_anomaly(True)
    env = gym.make(game, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 10 == 0,
        video_folder="bce-video-folder",
        name_prefix="video-",
    )
    device = 'cpu'

    replay_capacity = 4096

    policy_nn = MLPBCE(
        dim_input = 100800,
        dim_output = int(env.action_space.n)
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    episodes = 0
    learning_rate = torch.tensor(.0001, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)
    best_reward = -9999999999
    avg_reward = 0
    avg_loss = 0
    best_avg = -99999
    pbar = tqdm()
    while 1:
        n_eval = 10

        s, info = env.reset()
        s = torch.tensor(s, dtype=torch.float32).flatten().to(device)
        s_new = s
        s_old = s

        episode_over = False


        experience = []
        rewards = []

        actions = list(range(int(env.action_space.n)))


        n_frames =0
        while not episode_over:
            n_frames += 1
            with torch.no_grad():
                a = policy_nn(s)

            action = np.random.choice(actions, p=nn.Softmax()(a).cpu().detach().numpy())

            experience.append([s, action])
            s_old = s_new
            s_new, r, terminated, truncated, info = env.step(action)
            s_new = torch.tensor(s_new, dtype=torch.float32).flatten().to(device)
            s = s_new - s_old

            rewards.append(r)
            episode_over = terminated or truncated
            pbar.update()

        gs = []
        running_g = 0
        for R in rewards[::-1]:
            running_g = R + .99 * running_g
            gs.insert(0, running_g)

        loss = torch.tensor(0, dtype=torch.float32).to(device)

        ones = torch.tensor(1, dtype=torch.float32).to(device)
        for i, (s, action) in enumerate(experience):

            loss += -loss_fn(policy_nn(s)[action],ones)*gs[i]
            for _a in range(int(env.action_space.n)):
                loss += loss_fn(policy_nn(s)[_a], ones)*gs[i]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        best_reward = max(sum(rewards), best_reward)
        avg_reward += sum(rewards)/n_eval
        avg_loss += float(loss)/n_eval

        if (episodes % n_eval == 0):
            print()
            print(f"{avg_loss=}")
            print(f"{best_reward=}")
            print(f"{avg_reward=}")

            for p in  policy_nn.parameters():
                print(f"{torch.mean(torch.abs(p))=}")
                print(f"{learning_rate*torch.mean(torch.abs(p.grad))=}")
            if best_avg < avg_reward:
                torch.save(policy_nn.state_dict(), 'pong.pt')
                print("saved model")
                best_avg = avg_reward

            avg_reward = 0

        episodes += 1

    env.close()

if __name__ == "__main__":
    main()
