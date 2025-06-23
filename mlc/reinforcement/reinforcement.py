from torch import nn
from torch.optim import Adam
import torch
from math import ceil
from copy import deepcopy
import numpy as np
import time
from collections import deque
from utils import PreprocessFrame
from matplotlib import pyplot as plt

def train_qlearn(model, env, num_episodes=1000, max_steps = 10000, gamma=0.99, epsilon_decay=0.999, nframes = 1, skip_frames = 0, preprocess_frame = PreprocessFrame, render = False):
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
    buffer_size = 100000
    batch_size = 32
    
    state_shape = list(model.input_shape)
    state_shape[-1] = state_shape[-1]*nframes

    states_buffer = np.empty((buffer_size, *state_shape))
    actions_buffer = np.empty((buffer_size), dtype = np.int64)
    rewards_buffer = np.empty((buffer_size))
    dones_buffer = np.empty((buffer_size))

    rewards_history = []
    steps_history = []
    error_history = []

    epsilon = 1
    try:
        for e in range(num_episodes):
            state, info = env.reset()
            state = preprocess_frame(state)

            lives = info.get("lives", 0)

            previous_states = deque(maxlen=nframes)
            previous_states.append(state)
            for _ in range(nframes):
                state, *_ = env.step(0)
                state = preprocess_frame(state)
                previous_states.append(state)

            done = False

            total_reward = 0

            elapsed_time = 0
            total_action_time = 0 
            total_step_time = 0
            total_sample_time = 0
            total_q_time = 0
            total_train_time = 0

            td_errors = []
            start_time0 = time.perf_counter()
            for t in range(max_steps):
                if done:
                    break 

                start_time = time.perf_counter()
                # Choose action
                prev_states = np.concatenate(previous_states, axis = -1)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        states_tensor = torch.from_numpy(prev_states).float().unsqueeze(0)  # keep on CPU
                        q_values = cpu_model(states_tensor)
                        action = q_values.argmax(1).cpu().numpy()[0]  # Get the action with the highest Q-value
                        action = np.array(action, dtype=np.int64)

                action_time = time.perf_counter() 
                total_action_time += action_time - start_time

                # Take action
                next_state, reward, terminal, truncated, info = env.step(action)
                done = terminal or truncated
                for _ in range(skip_frames):
                    if done:
                        break
                    next_state, r, terminal, truncated, info = env.step(action)
                    done = terminal or truncated
                    reward += r

                reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
                if lives > info.get("lives", 0):
                    lives = info.get("lives", 0)
                    #reward += -10

                next_state = preprocess_frame(next_state)
                previous_states.append(next_state)
                total_reward += reward

                # Store transition in memory (move to GPU)
                idx = current_index % buffer_size
                states_buffer[idx] = prev_states
                actions_buffer[idx] = action
                rewards_buffer[idx] = reward
                dones_buffer[idx] = done
                current_index += 1

                if render:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(state)
                    plt.axis('off')
                    plt.savefig(f"frames/frame_{e}_{t}.png")
                    plt.close()

                step_time = time.perf_counter()
                total_step_time += step_time - action_time

                # Update state
                state = next_state

                # Train the model
                if current_index > 10000:
                    max_idx = min(current_index, buffer_size)
                                        
                    idx = np.random.choice(max_idx - 1, batch_size, replace=False)
                    
                    states_tensor = torch.from_numpy(states_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    actions_tensor = torch.from_numpy(actions_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    rewards_tensor = torch.from_numpy(rewards_buffer[idx]).pin_memory().to(device, non_blocking = True)
                    next_states_tensor = torch.from_numpy(states_buffer[idx+1]).pin_memory().to(device, non_blocking = True)
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

                    td_error = (q_values.float() - expected_q_values.float()).abs()
                    td_error = td_error.mean().detach().cpu().numpy()
                    td_errors.append(td_error)

                    # After each update, copy weights to cpu_model
                    train_time = time.perf_counter()
                    total_train_time += train_time - q_time

                    # Update target model
                    if current_index % 4000 == 0:
                        target_model.load_state_dict(model.state_dict())

                    cpu_model.load_state_dict(model.state_dict())

            # Append total reward to history
            error = np.mean(td_errors)
            error_history.append(error)
            rewards_history.append(total_reward)
            steps_history.append(t)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time0

            epsilon *= epsilon_decay
            if epsilon < 0.01:
                epsilon = 0.01
            print(f"Episode {e+1}/{num_episodes}, Steps in Episode: {t}, Total Reward: {total_reward}, TD Error: {error:.4f}, Total Steps: {current_index}, Epsilon: {epsilon:.4f}")
            print(f"Time Elapsed: {elapsed_time:.4f} seconds, Action Time: {total_action_time:.4f}, Step Time: {total_step_time:.4f}, Sample Time: {total_sample_time:.4f}, Q Time: {total_q_time:.4f}, Train Time: {total_train_time:.4f}")

        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    torch.cuda.empty_cache()
    return rewards_history, steps_history, error_history

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