from torch import nn
from torch.optim import Adam
import torch
from math import ceil
from copy import deepcopy
import numpy as np
import time
from collections import deque

class CNN(nn.Module):
    def __init__(self, num_actions, frame_count = 1, layer_channels = 16, input_shape=(96, 96, 3), lr = 1e-4, softmax = True):
        super(CNN, self).__init__()

        self.input_channels = input_shape[-1]
        self.input_size = input_shape[:-1]
        self.num_actions = num_actions
        self.layer_channels = layer_channels
        self.frame_count = frame_count

        kernel_size = 5
        conv_layers = [
            nn.Conv2d(self.input_channels * frame_count, layer_channels, bias = False, kernel_size = 9, stride=4, padding = 9//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels, layer_channels*2, bias = False, kernel_size = 5, padding = 5//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels*2, layer_channels*2, bias = False, kernel_size = 5, stride = 2, padding = 5//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels*2, layer_channels*4, bias = False, kernel_size = 3, padding = 3//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels*4, layer_channels*4, bias = False, kernel_size = 3, stride = 2, padding = 3//2),
            nn.LeakyReLU(0.01)
            ]
        
        a = [
            nn.Conv2d(self.input_channels * frame_count, layer_channels, bias = False, kernel_size = kernel_size, stride=2, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=1, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=2, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=1, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=2, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=1, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=2, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels),
            nn.Conv2d(layer_channels, layer_channels, kernel_size = kernel_size, bias = False, stride=1, padding = kernel_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(layer_channels)]
        
        mlp_layers = [
            nn.Flatten(),
            nn.Linear(layer_channels * 4 * ceil(self.input_size[0] / 2**4) * ceil(self.input_size[1] / 2**4), layer_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(layer_channels, num_actions)
        ]

        self.softmax = softmax
        if self.softmax:
            mlp_layers.append(nn.Softmax(dim = -1))
        self.q1 = nn.Sequential(*conv_layers)
        self.q2 = nn.Sequential(*mlp_layers)

        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float()
        x = self.q1(x)
        return self.q2(x)

def train_qlearn_cuda(model, env, num_episodes=1000, gamma=0.9, epsilon_decay=0.9):
    import torch
    import torch.nn as nn

    device = "cuda"

    # Create target model (on GPU)
    target_model = deepcopy(model)
    target_model.load_state_dict(model.state_dict())
    target_model.to(device)
    model.to(device)

    # Create CPU model for env interaction
    cpu_model = deepcopy(model)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.to("cpu")

    current_index = 0
    buffer_size = 4096
    state_shape = env.observation_space.shape

    # Keep replay buffer on GPU
    states_buffer = torch.zeros((buffer_size, *state_shape), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((buffer_size,), dtype=torch.int64, device=device)
    rewards_buffer = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
    rewards_history = []

    epsilon = 1
    try:
        for e in range(num_episodes):
            state, *_ = env.reset()
            state = torch.from_numpy(np.array(state)).float()  # keep on CPU for env
            done = False

            total_reward = 0

            elapsed_time = 0
            total_action_time = 0 
            total_step_time = 0
            total_sample_time = 0
            total_q_time = 0
            total_train_time = 0

            while not done:
                start_time = time.perf_counter()
                # Choose action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        states_tensor = state.unsqueeze(0)  # still on CPU
                        q_values = cpu_model(states_tensor)
                        action = torch.argmax(q_values, dim=1).item()

                action_time = time.perf_counter() 
                total_action_time += action_time - start_time

                # Take action
                next_state, reward, terminal, truncated, _ = env.step(action)
                next_state = torch.from_numpy(np.array(next_state)).float()  # keep on CPU
                done = terminal or truncated
                total_reward += reward

                # Store transition in memory (move to GPU)
                index = current_index % buffer_size
                states_buffer[index] = state.to(device)
                actions_buffer[index] = torch.tensor(action, dtype=torch.int64, device=device)
                rewards_buffer[index] = torch.tensor(reward, dtype=torch.float32, device=device)
                dones_buffer[index] = torch.tensor(done, dtype=torch.bool, device=device)
                current_index += 1

                step_time = time.perf_counter()
                total_step_time += step_time - action_time

                # Update state
                state = next_state

            # Append total reward to history
            rewards_history.append(total_reward)

            # Train the model
            if current_index > 1000:
                batch_size = 128
                for i in range(4 * current_index//batch_size):
                    step_time = time.perf_counter()
                    idx = np.random.choice(min(current_index, buffer_size) - 1, batch_size, replace=False)

                    # All buffers are already on GPU
                    states_tensor = states_buffer[idx]
                    actions_tensor = actions_buffer[idx]
                    rewards_tensor = rewards_buffer[idx]
                    next_states_tensor = states_buffer[idx+1]
                    dones_tensor = dones_buffer[idx]

                    sample_time = time.perf_counter()
                    total_sample_time += sample_time - step_time

                    # Double Q-Learning
                    q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                    # Action selection using online network
                    next_actions = model(next_states_tensor).argmax(1)
                    # Action evaluation using target network
                    next_q_values = target_model(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    expected_q_values = rewards_tensor + (gamma * next_q_values * (~dones_tensor))

                    q_time = time.perf_counter()
                    total_q_time += q_time - sample_time

                    loss = nn.MSELoss()(q_values.float(), expected_q_values.float())

                    model.optim.zero_grad()
                    loss.backward()
                    model.optim.step()

                    # After each update, copy weights to cpu_model
                    train_time = time.perf_counter()
                    total_train_time += train_time - q_time

                cpu_model.load_state_dict(model.state_dict())
                
            # Update target model
            if e % 10 == 0:
                target_model.load_state_dict(model.state_dict())
            
            end_time = time.perf_counter()
            elapsed_time += end_time - start_time

            epsilon *= epsilon_decay
            if epsilon < 0.01:
                epsilon = 0.01
            print(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward}, Time Elapsed: {elapsed_time} seconds")
            print(f"Action Time: {total_action_time:.4f}, Step Time: {total_step_time:.4f}, Sample Time: {total_sample_time:.4f}, Q Time: {total_q_time:.4f}, Train Time: {total_train_time:.4f}")

        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        torch.cuda.empty_cache()
        return rewards_history

    torch.cuda.empty_cache()
    return rewards_history

def train_qlearn(model, env, num_episodes=1000, gamma=0.9, epsilon_decay=0.9):
    import torch
    import torch.nn as nn

    device = "cuda"

    # Create target model (on GPU)
    target_model = deepcopy(model)
    target_model.load_state_dict(model.state_dict())
    target_model.to(device)
    model.to(device)

    # Create CPU model for env interaction
    cpu_model = deepcopy(model)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.to("cpu")

    current_index = 0
    buffer_size = 4096
    batch_size = 256
    state_shape = env.observation_space.shape

    states_buffer = np.empty((buffer_size, *state_shape))
    actions_buffer = np.empty((buffer_size), dtype = np.int64)
    rewards_buffer = np.empty((buffer_size))
    next_states_buffer = np.empty((buffer_size, *state_shape))
    dones_buffer = np.empty((buffer_size))

    rewards_history = []

    epsilon = 1
    try:
        for e in range(num_episodes):
            state, *_ = env.reset()
            done = False

            total_reward = 0

            elapsed_time = 0
            total_action_time = 0 
            total_step_time = 0
            total_sample_time = 0
            total_q_time = 0
            total_train_time = 0

            while not done:
                start_time = time.perf_counter()
                # Choose action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        states_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)  # keep on CPU
                        q_values = cpu_model(states_tensor)
                        action = np.argmax(q_values)

                action_time = time.perf_counter() 
                total_action_time += action_time - start_time

                # Take action
                next_state, reward, terminal, truncated, _ = env.step(action)
                done = terminal or truncated
                total_reward += reward

                # Store transition in memory (move to GPU)
                idx = current_index % buffer_size
                states_buffer[idx] = state
                actions_buffer[idx] = action
                rewards_buffer[idx] = reward
                next_states_buffer[idx] = next_state
                dones_buffer[idx] = done
                current_index += 1

                step_time = time.perf_counter()
                total_step_time += step_time - action_time

                # Update state
                state = next_state

            # Append total reward to history
            rewards_history.append(total_reward)

            # Train the model
            if current_index > 1000:
                max_idx = min(current_index, buffer_size)
                for i in range(max_idx//batch_size):
                    step_time = time.perf_counter()
                    
                    idx = np.random.choice(max_idx, batch_size, replace=False)
                    # All buffers are already on GPU
                    states_tensor = torch.from_numpy(states_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    actions_tensor = torch.from_numpy(actions_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    rewards_tensor = torch.from_numpy(rewards_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    next_states_tensor = torch.from_numpy(next_states_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    dones_tensor = torch.from_numpy(dones_buffer[idx]).pin_memory().to(device, non_blocking = True)

                    sample_time = time.perf_counter()
                    total_sample_time += sample_time - step_time

                    # Double Q-Learning
                    q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                    # Action selection using online network
                    next_actions = model(next_states_tensor).argmax(1)
                    # Action evaluation using target network
                    next_q_values = target_model(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    expected_q_values = rewards_tensor + (gamma * next_q_values * (dones_tensor))

                    q_time = time.perf_counter()
                    total_q_time += q_time - sample_time

                    loss = nn.MSELoss()(q_values.float(), expected_q_values.float())

                    model.optim.zero_grad()
                    loss.backward()
                    model.optim.step()

                    # After each update, copy weights to cpu_model
                    train_time = time.perf_counter()
                    total_train_time += train_time - q_time

                cpu_model.load_state_dict(model.state_dict())
                
            # Update target model
            if e % 10 == 0:
                target_model.load_state_dict(model.state_dict())
            
            end_time = time.perf_counter()
            elapsed_time += end_time - start_time

            epsilon *= epsilon_decay
            if epsilon < 0.01:
                epsilon = 0.01
            print(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward}, Time Elapsed: {elapsed_time} seconds")
            print(f"Action Time: {total_action_time:.4f}, Step Time: {total_step_time:.4f}, Sample Time: {total_sample_time:.4f}, Q Time: {total_q_time:.4f}, Train Time: {total_train_time:.4f}")

        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        torch.cuda.empty_cache()
        return rewards_history

    torch.cuda.empty_cache()
    return rewards_history

def train_reinforce_frames(model, env, nframes = 1, num_episodes=1000, gamma=0.9):
    ## Create target model
    target_model = deepcopy(model)
    target_model.load_state_dict(model.state_dict())

    device = next(model.parameters()).device
    target_model.to(device)

    rewards_history = []

    all_actions = list(range(env.action_space.n))
    try:
        for e in range(num_episodes):
            #buffer_size = 1024
            #replay_buffer = deque(maxlen=buffer_size)
            replay_buffer = []

            state, info = env.reset()
            lives = info["lives"]
            done = False

            # Initialize deque to hold previous states
            previous_states = deque(maxlen=nframes)
            previous_states.append(state)

            for _ in range(nframes - 1):
                # skip first frames, safe on *most* envs
                state, *_ = env.step(0)
                previous_states.append(state)

            total_reward = 0

            elapsed_time = 0
            total_action_time = 0 
            total_step_time = 0
            total_sample_time = 0
            total_q_time = 0
            total_train_time = 0

            max_episode_len = 10000
            for t in range(max_episode_len):
                start_time = time.perf_counter()
                # Choose action
                with torch.no_grad():
                    prev_states = np.concatenate(previous_states, axis = -1)
                    state_tensor = torch.from_numpy(prev_states).unsqueeze(0).to(device)
                    action_dist = model(state_tensor)
                    action_dist = action_dist.squeeze(0).cpu().numpy()
                    action = np.random.choice(all_actions, p=action_dist)
                
                action_time = time.perf_counter() 
                total_action_time += action_time - start_time

                # Take action
                next_state, reward, terminal, truncated, info = env.step(action)
                if lives > info["lives"]:
                    lives = info["lives"]
                    reward -= 10
                total_reward += reward

                replay_buffer.append([prev_states, action, reward])

                step_time = time.perf_counter()
                total_step_time += step_time - action_time

                # Update state
                #state = next_state
                previous_states.append(next_state)

                done = terminal or truncated
                if done:
                    break

            running_rewards = 0
            for i in range(len(replay_buffer)):
                r = replay_buffer[-i-1][2]
                running_rewards = r + gamma * running_rewards
                replay_buffer[-i-1][2] = running_rewards

            sample_time = time.perf_counter()
            total_sample_time += sample_time - step_time
            
            states_tensor = torch.tensor(np.array([t[0] for t in replay_buffer]), dtype=torch.float32, device=device)
            preds = model(states_tensor)
            
            q_time = time.perf_counter()
            total_q_time += q_time - sample_time

            actions = torch.tensor(np.array([t[1] for t in replay_buffer]), dtype=torch.long, device=device)
            discounted_rewards = torch.tensor(np.array([t[2] for t in replay_buffer]), dtype=torch.float32, device=device)
            log_probs = torch.log(preds.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
            loss = -(log_probs * discounted_rewards).sum()

            model.optim.zero_grad()
            loss.backward()
            model.optim.step()

            train_time = time.perf_counter()
            total_train_time += train_time - q_time
            
            end_time = time.perf_counter()
            elapsed_time += end_time - start_time

            # Append total reward to history
            rewards_history.append(total_reward)

            print(f"Episode {e+1}/{num_episodes}, Steps in Episode: {t}, Total Reward: {total_reward}, Time Elapsed: {elapsed_time} seconds")
            print(f"Action Time: {total_action_time:.4f}, Step Time: {total_step_time:.4f}, Sample Time: {total_sample_time:.4f}, Q Time: {total_q_time:.4f}, Train Time: {total_train_time:.4f}")
        
        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return rewards_history
    
    return rewards_history