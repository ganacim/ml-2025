import time
import numpy as np
import gymnasium as gym
import ale_py

def test_env(env):
    env.reset()
    secs = []
    try:
        actions = [3]
        total_reward = 0
        #for i in range(1000):
        done = False
        while not done:
            start = time.perf_counter()
            action = env.action_space.sample()
            #action = actions[i % len(actions)]
            #if i > len(actions):
            #    action = 0
            #action = int(input(f"{i}:"))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                env.reset()
            end = time.perf_counter()
            #print(f"Step took {end - start:.4f} seconds, reward: {reward}, action: {action}")
            secs.append(end - start)
    except KeyboardInterrupt:
        pass
    print(obs.shape, type(obs))
    print(f"Average step time: {np.mean(secs):.4f}+-{np.std(secs)} seconds")
    print(f"Total reward: {total_reward}")
    env.close()


env = gym.make("ALE/Riverraid-v5", continuous = False, render_mode="human")
#env = gym.make("ALE/Pacman-v5", continuous = False, render_mode="human")
test_env(env)