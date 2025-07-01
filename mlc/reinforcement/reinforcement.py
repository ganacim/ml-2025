from torch import nn
from torch.optim import Adam
import torch
from math import ceil
from copy import deepcopy
import numpy as np
import time
from collections import deque
from utils import PreprocessFrame, Memory, soft_update, OrnsteinUhlenbeck
from matplotlib import pyplot as plt
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import os

def train_tdddpg(env, actor, critic, episodes=200, max_steps=100000, start_steps = 1000, noise_mult = 1, buffer_size = 100000, nframes = 1, gamma=0.99, BATCH_SIZE=256, verbose=20, checkpoint = 50, patience = 50, policy_delay = 2, sigma = 0.1, alpha = 0.5, beta = 0.5, tau = 0.005):
   
    replay_memory = Memory(state_dims=env.states, action_dims=env.actions, size = buffer_size, alpha=alpha)

    reward_hist = []
    step_hist = []
    step_acc = []
    actor_losses = []
    critic_losses = []
    td_errors = []

    timestamp = datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
    env_name = env.unwrapped.spec.id
    output_folder = f"agents/td3/{env_name.replace('/', '_')}/{timestamp}"
    writer = SummaryWriter(output_folder + "/tensorboard")
    if not os.path.exists(output_folder + "/models"):
        os.makedirs(output_folder + "/models")

    device = next(actor.parameters()).device

    # Create target networks
    target_actor = deepcopy(actor)
    target_critic = deepcopy(critic)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    ou_noise = OrnsteinUhlenbeck(env.action_space.shape)
    start_time0 = time.perf_counter()
    try:
        total_steps = 0
        total_reward = 0
        best_reward = 0
        patience_counter = 0
        for e in range(episodes):
            # Reset episode
            td_err = []
            obs, *_ = env.reset()
            previous_states = deque(maxlen = nframes)
            for _ in range(nframes):
                previous_states.append(obs)
            prev_state = np.concat(previous_states, axis = -1)
            ou_noise.reset()

            if total_reward > 305:
                break
            total_reward = 0

            start_time = time.perf_counter()
            actor_loss_temp = []
            critic_loss_temp = []
            for t in range(max_steps):
                # Sample a from u(s, theta) + Normal
                if BATCH_SIZE and total_steps < start_steps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(prev_state, device=device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                        action = actor(obs_tensor).detach().cpu().squeeze().numpy()  # Get action from actor network
                        action = action + noise_mult*ou_noise.sample() #np.random.normal(0, sigma)  # Noise = Normal(0,sigma)?
                        action = np.clip(action, env.action_space.low, env.action_space.high)

                # Take action, observe s_p and r
                next_obs, reward, terminal, truncated, info = env.step(action)
                previous_states.append(next_obs)
                next_state = np.concat(previous_states, axis = -1)
                total_reward += reward

                # Save to memory
                replay_memory.add(prev_state, action, reward, next_state, terminal)
                prev_state = next_state

                # Train minibatch
                if BATCH_SIZE and total_steps >= start_steps:
                    samples, weights, idx = replay_memory.sample(BATCH_SIZE, beta)
                    beta = min(1, beta + 1e-4)

                    states, actions, rewards, next_states, dones, = samples

                    weights = torch.tensor(weights, device=device, dtype=torch.float32)
                    states = torch.tensor(states, device=device, dtype=torch.float32)
                    actions = torch.tensor(actions, device=device, dtype=torch.float32)
                    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
                    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
                    dones = torch.tensor(dones, device=device, dtype=torch.float32)

                    with torch.no_grad():
                        c = torch.cat((next_states, target_actor(next_states)), dim=1)
                        target_q = rewards + gamma * (1 - dones) * torch.min(*target_critic(c))

                    # Update critic
                    c2 = torch.cat((states, actions), dim=1)

                    current_q1, current_q2 = critic(c2)
                    td_error1 =  current_q1 - target_q
                    td_error2 =  current_q2 - target_q
                    critic_loss = (weights * td_error1.pow(2)).mean() + (weights * td_error2.pow(2)).mean()

                    #critic_loss = nn.MSELoss()(critic(c2), q)
                    critic.optim.zero_grad()
                    critic_loss.backward()
                    critic.optim.step()
                    critic_loss_temp.append(critic_loss.item())

                    #td_error = (td_error1.abs()/2 + td_error2.abs()/2).detach().cpu().numpy()
                    td_error = td_error1.detach().abs().cpu().numpy()
                    td_err.append(np.mean(td_error))
                    replay_memory.update_priorities(idx, td_error)

                    # Update actor
                    if policy_delay and total_steps % policy_delay == 0:
                        c_actor = torch.cat((states, actor(states)), dim=1)
                        actor_loss = -critic.c1(c_actor).mean()
                        actor.optim.zero_grad()
                        actor_loss.backward()
                        actor.optim.step()
                        actor_loss_temp.append(actor_loss.item())

                        # Update targets
                        soft_update(target_actor, actor, tau)
                        soft_update(target_critic, critic, tau)

                    #if total_steps % 200 == 0:
                    #    target_actor.load_state_dict(actor.state_dict())
                    #    target_critic.load_state_dict(critic.state_dict())

                #obs = next_obs
                total_steps += 1
                if terminal or truncated:
                    break

            reward_hist.append(total_reward)
            step_hist.append(t)
            step_acc.append(total_steps)

            patience_counter += 1
            if total_steps > start_steps:
                patience_counter = 0
                actor_losses.append(np.mean(actor_loss_temp))
                critic_losses.append(np.mean(critic_loss_temp))
                td_errors.append(np.mean(td_err))
            else:
                actor_losses.append(0)
                critic_losses.append(0)
                td_errors.append(0)

            # TensorBoard logging
            writer.add_scalar("Reward/Episode", total_reward, e)
            writer.add_scalar("Steps/Episode", t, e)
            writer.add_scalar("Loss/Actor", actor_losses[-1], e)
            writer.add_scalar("Loss/Critic", critic_losses[-1], e)

            # Periodic model backup
            if total_reward > best_reward:
                best_reward = total_reward
                try:
                    os.reanme(f"{output_folder + "/models"}/actor_best.pth", f"{output_folder + "/models"}/actor_best_old.pth")
                    os.reanme(f"{output_folder + "/models"}/critic_best.pth", f"{output_folder + "/models"}/critic_best_old.pth")
                except:
                    pass
                torch.save(actor.state_dict(), f"{output_folder + "/models"}/actor_best.pth")
                torch.save(critic.state_dict(), f"{output_folder + "/models"}/critic_best.pth")

            if checkpoint and (e + 1) % checkpoint == 0:
                torch.save(actor.state_dict(), f"{output_folder + "/models"}/actor_ep{e+1}.pth")
                torch.save(critic.state_dict(), f"{output_folder + "/models"}/critic_ep{e+1}.pth")

            end_time = time.perf_counter()
            s = end_time - start_time
            if verbose and e % verbose == 0:
                print(f"Episode {e} finished with reward {total_reward} in {t} steps and {s:4f} time, Total steps: {total_steps}")
                print(f"Actor loss: {actor_losses[-1]}, Critic loss: {critic_losses[-1]}, Total steps: {total_steps}")
            
            if patience_counter > patience:
                break

    except KeyboardInterrupt:
        print("Interrupting...")
    finally:
        writer.close()
    
    # Save training data and model after finishing
    torch.save(actor.state_dict(), f"{output_folder + "/models"}/actor_ep{e+1}.pth")
    torch.save(critic.state_dict(), f"{output_folder + "/models"}/critic_ep{e+1}.pth")

    end_time = time.perf_counter()
    delta_time = end_time - start_time0
    avg = delta_time / (e if 'e' in locals() else 1)
    print(f"Training finished after {e if 'e' in locals() else 0} episodes and {delta_time:4f} seconds, averaging {avg:4f} seconds per episode")
    return reward_hist, step_hist, step_acc, actor_losses, critic_losses, td_errors

def train_ddpg(env, actor, critic, episodes=200, max_steps=10000, noise_mult = 1, start_steps = 1000, gamma=0.99, BATCH_SIZE=256, verbose=20, checkpoint = 50, sigma = 0.1, alpha = 0.5, beta = 0.5):
    try:
        replay_memory = Memory(state_dims=env.states, action_dims=env.actions, alpha=alpha)
    except:
        replay_memory = Memory(state_dims=env.states, action_dims=env.actions, size = 10000, alpha=alpha)

    reward_hist = []
    step_hist = []
    actor_losses = []
    critic_losses = []

    timestamp = datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
    env_name = env.unwrapped.spec.id
    output_folder = f"agents/ddpg/{env_name.replace('/', '_')}/{timestamp}"
    writer = SummaryWriter(output_folder + "/tensorboard")
    if not os.path.exists(output_folder + "/models"):
        os.makedirs(output_folder + "/models")

    device = next(actor.parameters()).device

    # Create target networks
    target_actor = deepcopy(actor)
    target_critic = deepcopy(critic)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    start_time0 = time.perf_counter()
    try:
        total_steps = 0
        for e in range(episodes):
            # Reset episode
            obs, *_ = env.reset()
            total_reward = 0

            start_time = time.perf_counter()
            actor_loss_temp = []
            critic_loss_temp = []
            for t in range(max_steps):
                # Sample a from u(s, theta) + Normal
                if BATCH_SIZE and total_steps < start_steps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                        action = actor(obs_tensor).detach().cpu().squeeze().numpy()  # Get action from actor network
                        action = action + noise_mult * np.random.normal(0, sigma)  # Noise = Normal(0,sigma)?
                        action = np.clip(action, env.action_space.low, env.action_space.high)

                # Take action, observe s_p and r
                next_obs, reward, terminal, truncated, info = env.step(action)
                total_reward += reward

                # Save to memory
                replay_memory.add(obs, action, reward, next_obs, terminal)

                # Train minibatch
                if BATCH_SIZE and total_steps >= start_steps:
                    samples, weights, idx = replay_memory.sample(BATCH_SIZE, beta)
                    states, actions, rewards, next_states, dones, = samples

                    weights = torch.tensor(weights, device=device, dtype=torch.float32)
                    states = torch.tensor(states, device=device, dtype=torch.float32)
                    actions = torch.tensor(actions, device=device, dtype=torch.float32)
                    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
                    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
                    dones = torch.tensor(dones, device=device, dtype=torch.float32)

                    with torch.no_grad():
                        c = torch.cat((next_states, target_actor(next_states)), dim=1)
                        q = rewards + gamma * (1 - dones) * target_critic(c)

                    # Update critic
                    c2 = torch.cat((states, actions), dim=1)

                    td_error = critic(c2) - q
                    critic_loss = (weights * td_error.pow(2)).mean()

                    #critic_loss = nn.MSELoss()(critic(c2), q)
                    critic.optim.zero_grad()
                    critic_loss.backward()
                    critic.optim.step()
                    critic_loss_temp.append(critic_loss.item())

                    # Update actor
                    c_actor = torch.cat((states, actor(states)), dim=1)
                    actor_loss = -critic(c_actor).mean()
                    actor.optim.zero_grad()
                    actor_loss.backward()
                    actor.optim.step()
                    actor_loss_temp.append(actor_loss.item())

                    td_error = td_error.detach().cpu().numpy()
                    replay_memory.update_priorities(idx, td_error)
                    # Update targets
                    #soft_update(target_actor, actor, tau)
                    #soft_update(target_critic, critic, tau)

                    if total_steps % 200 == 0:
                        target_actor.load_state_dict(actor.state_dict())
                        target_critic.load_state_dict(critic.state_dict())

                obs = next_obs
                total_steps += 1
                if terminal or truncated:
                    break

            reward_hist.append(total_reward)
            step_hist.append(t)
            if total_steps > start_steps:
                actor_losses.append(np.mean(actor_loss_temp))
                critic_losses.append(np.mean(critic_loss_temp))
            else:
                actor_losses.append(0)
                critic_losses.append(0)

            # TensorBoard logging
            writer.add_scalar("Reward/Episode", total_reward, e)
            writer.add_scalar("Steps/Episode", t, e)
            writer.add_scalar("Loss/Actor", actor_losses[-1], e)
            writer.add_scalar("Loss/Critic", critic_losses[-1], e)

            # Periodic model backup
            if checkpoint and (e + 1) % checkpoint == 0:
                torch.save(actor.state_dict(), f"{output_folder + "/models"}/actor_ep{e+1}.pth")
                torch.save(critic.state_dict(), f"{output_folder + "/models"}/critic_ep{e+1}.pth")

            end_time = time.perf_counter()
            s = end_time - start_time
            beta = min(1, beta + 0.01)
            if verbose and e % verbose == 0:
                print(f"Episode {e} finished with reward {total_reward} in {t} steps and {s:4f} time, Total steps: {total_steps}")
                print(f"Actor loss: {actor_losses[-1]}, Critic loss: {critic_losses[-1]}, Total steps: {total_steps}")

    except KeyboardInterrupt:
        print("Interrupting...")
    finally:
        writer.close()
    
    # Save training data and model after finishing
    torch.save(actor.state_dict(), f"{output_folder + "/models"}/actor_ep{e+1}.pth")
    torch.save(critic.state_dict(), f"{output_folder + "/models"}/critic_ep{e+1}.pth")

    end_time = time.perf_counter()
    delta_time = end_time - start_time0
    avg = delta_time / (e if 'e' in locals() else 1)
    print(f"Training finished after {e if 'e' in locals() else 0} episodes and {delta_time:4f} seconds, averaging {avg:4f} seconds per episode")
    return reward_hist, step_hist, actor_losses, critic_losses

def train_qlearn(model, env, num_episodes=1000, max_steps = 10000, gamma=0.99, epsilon = 1, epsilon_decay=0.999, nframes = 1, skip_frames = 0, preprocess_frame = PreprocessFrame, render = False):
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
    buffer_size = 10000
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
                for _ in range(skip_frames):
                    if terminal or truncated:
                        break
                    next_state, r, terminal, truncated, info = env.step(action)
                    reward += r

                reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
                if lives > info.get("lives", 0):
                    lives = info.get("lives", 0)
                    #reward = -10

                next_state = preprocess_frame(next_state)
                previous_states.append(next_state)
                total_reward += reward

                # Store transition in memory (move to GPU)
                idx = current_index % buffer_size
                states_buffer[idx] = prev_states
                actions_buffer[idx] = action
                rewards_buffer[idx] = reward
                dones_buffer[idx] = terminal
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
                if current_index > 1000:
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
                    if current_index % 1000 == 0:
                        target_model.load_state_dict(model.state_dict())

                    cpu_model.load_state_dict(model.state_dict())
                
                if terminal or truncated:
                    break

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