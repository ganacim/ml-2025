import time
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import torch
from reinforcement import CNN

def test_env(env, model = None):
    state, *_ = env.reset()
    
    nframes = 1
    if model:
        nframes = model.frame_count
        device = next(model.parameters()).device

    # Initialize deque to hold previous states
    previous_states = deque(maxlen=nframes)
    previous_states.append(state)
        
    for _ in range(nframes - 1):
        # skip first frames, safe on *most* envs
        state, *_ = env.step(0)
        previous_states.append(state)
        

    secs = []
    try:
        total_reward = 0
        done = False

        all_actions = list(range(env.action_space.n))
        while not done:
            start = time.perf_counter()

            action = env.action_space.sample()
            if model:
                with torch.no_grad():
                    prev_states = np.concatenate(previous_states, axis = -1)
                    state_tensor = torch.from_numpy(prev_states).unsqueeze(0).to(device)
                    action_dist = model(state_tensor)
                    action_dist = action_dist.squeeze(0).cpu().numpy()
                    action = np.random.choice(all_actions, p=action_dist)
                    #action = np.argmax(action_dist)
            
            state, reward, terminated, truncated, info = env.step(action)
            previous_states.append(state)
            print(info)

            total_reward += reward
            done = terminated or truncated
            if done:
                env.reset()

            end = time.perf_counter()

            secs.append(end - start)
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print(state.shape, type(state))
    print(f"Average step time: {np.mean(secs):.4f}+-{np.std(secs)} seconds")
    print(f"Total reward: {total_reward}")
    env.close()


#env = gym.make("ALE/Riverraid-v5", continuous = False, render_mode="human")
env = gym.make("ALE/Pacman-v5", continuous = False, render_mode="human")
#env = gym.make("CarRacing-v3", continuous = False)

#model = CNN(env.action_space.shape)
device = "cpu"
print(env.observation_space, env.action_space)
state_shape = env.observation_space.shape
action_shape = 5 #env.action_space.shape

model = CNN(action_shape, frame_count=4, layer_channels=32, input_shape = state_shape).to(device)
model.load_state_dict(torch.load("C:/Users/hss19/Documents/GitHub/ml-2025/notebooks/model.pth"))

test_env(env, model)