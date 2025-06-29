# train_luna.py

import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import re
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from collections import deque
import os

# Import the Base class and the new MLP network
from mlc.command.base import Base
from mlc.reinforcement.networks_luna import MLP


# A utility function to create a unique folder for each run
def get_time_as_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# The Train class inherits from Base to be discoverable by the mlc tool
class Train(Base):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.device = "cpu"
        if hparams["device"].startswith("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device(hparams["device"])
                print("CUDA is available! Training on GPU.")
            else:
                print("CUDA not available, falling back to CPU.")
        else:
            print("Training on CPU.")

        self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/mlp_agent/{get_time_as_str()}"
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Outputting results to: {self.output_folder}")
        self.writer = SummaryWriter(self.output_folder + "/tensorboard")

    @classmethod
    def name(cls):
        return "mlp_agent.train_luna"

    @staticmethod
    def add_arguments(parser):
        def _parse_device_arg(arg_value):
            pattern = re.compile(r"(cpu|cuda|cuda:\d+)")
            if not pattern.match(arg_value):
                raise argparse.ArgumentTypeError("invalid value")
            return arg_value
        
        parser.add_argument("-g", "--game", default="LunarLander-v3", help="Environment name")
        parser.add_argument("--num_envs", default=8, type=int, help="Number of parallel environments")
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="Device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
        parser.add_argument("-v", "--video", type=int, default=5000, help="Save a permanent .mp4 video every n episodes")
        parser.add_argument("--tensorboard-video", type=int, default=400, help="Log a temporary video to TensorBoard every n episodes")
        # --- NEW ARGUMENT TO ENABLE WIND ---
        parser.add_argument("--enable-wind", action="store_true", help="Enable wind in the LunarLander environment.")


    def run(self):
        """Main training loop that runs indefinitely until interrupted."""
        game = self.hparams["game"]
        num_envs = self.hparams["num_envs"]
        device = self.device
        
        # Check if wind should be enabled from the new argument
        enable_wind = self.hparams.get("enable_wind", False)
        if enable_wind:
            print("Wind is ENABLED for this training run.")

        # --- UPDATED ENVIRONMENT CREATION ---
        # The lambda function now passes the enable_wind parameter to gym.make
        envs = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(game, enable_wind=enable_wind) for _ in range(num_envs)]
        )

        s, _ = envs.reset()
        n_actions = int(envs.action_space[0].n)
        dim_input = s.shape[-1]

        policy_nn = MLP(dim_input=dim_input, dim_output=n_actions, dim_hidden=64).to(device)
        optimizer = torch.optim.Adam(policy_nn.parameters(), lr=self.hparams["learning_rate"])
        pbar = tqdm(desc="Training Episodes")

        states, _ = envs.reset()
        states = torch.tensor(states, dtype=torch.float32).to(device)

        replay_buffers = [deque(maxlen=4096) for _ in range(num_envs)]
        episode_start = np.zeros(envs.num_envs, dtype=bool)
        n_episodes = 0
        
        try:
            while True:
                with torch.no_grad():
                    action_dist = policy_nn(states)
                
                actions = action_dist.multinomial(num_samples=1).squeeze(-1).cpu().numpy()
                next_states, rewards, terminations, truncations, _ = envs.step(actions)

                # --- REWARD SHAPING (Optional): Add a time penalty ---
                # rewards -= 0.1 

                for i in range(envs.num_envs):
                    if not episode_start[i]:
                        replay_buffers[i].append({"state": states[i], "action": int(actions[i]), "reward": float(rewards[i])})

                states = torch.tensor(next_states, dtype=torch.float32).to(device)
                episode_start = np.logical_or(terminations, truncations)

                for i in range(envs.num_envs):
                    if episode_start[i]:
                        n_episodes += 1
                        pbar.update(1)

                        replay = list(replay_buffers[i])
                        replay_buffers[i].clear()

                        if not replay: continue

                        episode_duration = len(replay)
                        episode_rewards = [x["reward"] for x in replay]
                        
                        propagated_rewards = []
                        running_reward = 0
                        for R in episode_rewards[::-1]:
                            running_reward = R + 0.99 * running_reward
                            propagated_rewards.insert(0, running_reward)
                        
                        propagated_rewards = torch.tensor(propagated_rewards, dtype=torch.float32).to(device)
                        propagated_rewards = (propagated_rewards - propagated_rewards.mean()) / (propagated_rewards.std() + 1e-9)

                        replay_states = torch.stack([x["state"] for x in replay])
                        replay_actions = torch.tensor([x["action"] for x in replay], dtype=torch.int64).to(device)
                        
                        probs = policy_nn(replay_states)
                        log_probs = torch.log(probs + 1e-9)
                        
                        selected_log_probs = log_probs[range(len(replay_actions)), replay_actions]
                        loss = -(selected_log_probs * propagated_rewards).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), max_norm=1.0)
                        
                        optimizer.step()

                        self.writer.add_scalar('reward', sum(episode_rewards), n_episodes)
                        self.writer.add_scalar('loss', loss.item(), n_episodes)
                        self.writer.add_scalar('episode_duration', episode_duration, n_episodes)

                        # Logic for saving videos
                        save_mp4 = False
                        log_tb = False

                        if n_episodes == 1 or n_episodes == 400:
                            save_mp4 = True
                            log_tb = True
                        if n_episodes > 1 and n_episodes % self.hparams["video"] == 0:
                            save_mp4 = True
                        if n_episodes > 1 and n_episodes % self.hparams["tensorboard_video"] == 0:
                            log_tb = True
                        
                        if save_mp4:
                            self.save_mp4_video(game, policy_nn, n_episodes)
                        if log_tb:
                            self.log_tensorboard_video(game, policy_nn, n_episodes)
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")

        finally:
            pbar.close()
            envs.close()
            self.writer.close()
            print("Resources cleaned up. Exiting.")


    def save_mp4_video(self, game_name, policy, episode_num):
        video_folder = os.path.join(self.output_folder, "videos")
        video_env = RecordVideo(
            gym.make(game_name, render_mode="rgb_array", enable_wind=self.hparams.get("enable_wind", False)),
            video_folder=video_folder,
            episode_trigger=lambda x: x == 0,
            name_prefix=f"episode-{episode_num}"
        )
        self.run_single_episode(video_env, policy)
        print(f"\nSaved .mp4 video for episode {episode_num}.")
    
    def log_tensorboard_video(self, game_name, policy, episode_num):
        temp_env = gym.make(game_name, render_mode="rgb_array", enable_wind=self.hparams.get("enable_wind", False))
        frames = self.run_single_episode(temp_env, policy, collect_frames=True)
        
        frames_tensor = np.array(frames)
        frames_tensor = np.expand_dims(frames_tensor, axis=0)
        frames_tensor = np.transpose(frames_tensor, (0, 1, 4, 2, 3))
        self.writer.add_video('gameplay_progress', frames_tensor, episode_num, fps=30)

    def run_single_episode(self, env, policy, collect_frames=False):
        frames = []
        s, _ = env.reset()
        done = False
        while not done:
            if collect_frames:
                frames.append(env.render())
            with torch.no_grad():
                s_tensor = torch.tensor(s, dtype=torch.float32).to(self.device)
                action_dist = policy(s_tensor)
                action = torch.argmax(action_dist).item()
            s, _, term, trunc, _ = env.step(action)
            done = term or trunc
        env.close()
        return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Train.add_arguments(parser)
    args = parser.parse_args()
    hparams = vars(args)
    
    trainer = Train(hparams)
    trainer.run()
