import argparse
import gymnasium as gym
from time import sleep
import random
import json
import torch
from torch import nn
from mlc.reinforcement.network import MLP
import numpy as np
from tqdm import tqdm
import ale_py
from pathlib import Path


gym.register_envs(ale_py)

def main():
    game ="ALE/Pong-v5"
    torch.autograd.set_detect_anomaly(True)
    env = gym.make(game, render_mode="rgb_array")

    f = Path("out.jsonl").open("w")

    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 10 == 0,
        video_folder="video-folder-rivotril",
        name_prefix="video",
    )
    device = 'cpu'


    policy_nn = MLP(
        dim_input = 100_800,
        dim_output = int(env.action_space.n)
    ).to(device)

    policy_nn.load_state_dict(torch.load('pong-11640.pt', weights_only=True))
    policy_nn = policy_nn.to(device)

    episode = 0
    learning_rate = torch.tensor(0.003, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)
    best_reward = -9999999999
    avg_reward = 0
    avg_loss = 0
    nothing = 0
    best_avg = -99999
    pbar = tqdm()
    sigmoid = nn.Softmax()
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
            s = s / 255

            with torch.no_grad():
                a = policy_nn(s)

            action = np.random.choice(actions, p=a.cpu().detach().numpy())
            experience.append([s, action])

            s_old = s_new
            s_new, r, terminated, truncated, info = env.step(action)

            # reward doing nothing (rivotril)
            r += .03 * (action == 0)

            s_new = torch.tensor(s_new, dtype=torch.float32).flatten().to(device)
            s = s_new - s_old


            if action==0:
                nothing += 1

            rewards.append(r)
            episode_over = terminated or truncated

        pbar.update()

        gs = []
        running_g = 0
        for R in rewards[::-1]:
            if abs(R)>= .9:
                running_g = 0
            running_g = R + .99 * running_g
            gs.insert(0, running_g)

            # remove rivotril from propagation
            if abs(R) < .5:
                running_g -= R

        tgs = torch.tensor(gs)
        n_nonzero_rewards = sum([abs(x) > .9 for x in rewards])


        loss = torch.tensor(0, dtype=torch.float32).to(device)

        ts = torch.stack([s for s, _ in experience])
        preds = policy_nn(ts)
        for i, (s, action) in enumerate(experience):
            loss += -torch.log(preds[i][action])*tgs[i]/n_nonzero_rewards

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_reward = max(sum(rewards), best_reward)
        avg_reward += sum(rewards)/n_eval
        avg_loss += float(loss)/n_eval

        if (episode % n_eval == 0) and episode != 0:
            print()
            print()
            print(f"[ep {episode}] {avg_loss=}")
            print(f"[ep {episode}] {best_reward=}")
            print(f"[ep {episode}] {avg_reward=}")
            print(f"[ep {episode}] nothing={nothing/len(experience)}")

            out = {}
            out["episode"] = episode
            out["avg_loss"] = avg_loss
            out["best_reward"] = best_reward
            out["avg_reward"] = avg_reward
            out["avg_nothing"] = nothing/len(experience)
            f.write(json.dumps(out)+"\n")
            f.flush()

            #for p in  policy_nn.parameters():
            #    print(f"  {torch.mean(torch.abs(p))=}")
            #    print(f"  {learning_rate*torch.mean(torch.abs(p.grad))=}")
            if (best_avg < avg_reward) and episode != 0:
                torch.save(policy_nn.state_dict(), f'pong-rivotril-{episode}.pt')
                print()
                print("  saved model")
                best_avg = avg_reward

            avg_reward = 0
            avg_loss = 0
            nothing = 0

        episode += 1

    env.close()

if __name__ == "__main__":
    main()
