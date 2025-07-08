import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from time import sleep
from pathlib import Path   
import re
import random
import torch
from torch import nn
from mlc.reinforcement.networks import MLP
from mlc.reinforcement.networks import CNN
import numpy as np
from tqdm import tqdm
import ale_py
from torch.utils.tensorboard.writer import SummaryWriter

from collections import deque



from mlc.command.base import Base
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
        gym.register_envs(ale_py)
        def load_checkpoint(self, path, policy, optim, device):
            ckpt = torch.load(path, map_location=device)
            policy.load_state_dict(ckpt["model_state"])
            optim.load_state_dict(ckpt["optim_state"])
            return ckpt["episode"]
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
        parser.add_argument("-e", "--max_episodes", type=int, default=100000)

        parser.add_argument("-g", "--game", default="ALE/Pong-v5")
        parser.add_argument("--num_envs", default=4, type=int)
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
        #parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=100, help="check point every n episodes")
        parser.add_argument("-v", "--video", type=int, default=10, help="create a video every n episodes")
        parser.add_argument("-p", "--personal", action="store_true", help="enable personal folder")
        parser.set_defaults(personal=False)
        parser.add_argument("-n", "--name", type=str, default=None, help="name this run")

    def run(self):
        game = self.hparams["game"]
        #torch.autograd.set_detect_anomaly(True)
        num_envs = self.hparams["num_envs"]
        #envs = gym.make_vec(game, render_mode=None, vectorization_mode="async", num_envs=num_envs)

        envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(game) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)

        device = 'cpu'

        s, _ = envs.reset()
        s = torch.tensor(s, dtype=torch.float32).flatten(start_dim=1).to(device)

        n_actions = int(envs.action_space[0].n)
        all_actions = list(range(n_actions))

        SCHEDULE = {
    "entropy_weight": [(0, 0.1),  ( 1_500, 0.1), ( 2_500, 0.08), (10_000, 0.05),(60_000, 0.02), (100000, 0.01)],
    "lambda_still":   [(0, 1e-3), (25_000, 7e-4), (50_000, 5e-4)],
    "combo_bonus":    [(0, 1.0),  (15_000, 2.0),  (35_000, 3.0)],
    "move_bonus":     [(0, 3e-5), (20_000, 1e-5), (40_000, 0.0)],
    "beta_bonus":     [(0, 1e-3), ( 8_000, 2e-3)],
    "noise_sigma":  [(0, 0.05), (300, 0.001), (5_000,0), (15_000, 0.001), (15_050, 0), (25_000, 0.001), (25_050, 0) , (35_000, 0.001), (35_050, 0)  ]}


# RESET -------------------------------------------------------------
                # RESET -------------------------------------------------------------
        states, info0 = envs.reset()                    

        # vidas
        prev_lives = (np.array(info0["lives"])
                    if isinstance(info0, dict)
                    else np.array([inf["lives"] for inf in info0]))

        game_name = self.hparams["game"]

        if game_name == "ALE/Breakout-v5":
            states_uint8 = torch.from_numpy(states)                 
            states_img   = states_uint8.permute(0,3,1,2).float().div_(255) 
            obs_shape    = states_img.shape[1:]                   
            policy_nn    = CNN(obs_shape, n_actions).to(device)
            states_new   = states_img                               
            reset_on_big_reward = False
            norm_factor  = lambda nz: 1.0
        else:  # Pong
            states_new   = torch.from_numpy(states).float().flatten(start_dim=1).div_(255).to(device)
            policy_nn    = MLP(dim_input=states_new.shape[-1], dim_output=n_actions).to(device)
            reset_on_big_reward = True
            norm_factor  = lambda nz: max(nz, 1)



        states_old = states_new.clone()
        states      = states_new - states_old       

        learning_rate = torch.tensor(self.hparams["learning_rate"],
                                    dtype=torch.float32).to(device)
        optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)
        baseline = torch.tensor(0.0, device=device)


        ckpt_path = Path("meu_ckpt.pt")
        n_episodes = 0           

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            policy_nn.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            baseline = ckpt.get("baseline", baseline)
            n_episodes = ckpt["episode"]          
            print(f"Retomado do episódio {n_episodes}")


        pbar = tqdm(initial=n_episodes)

        states_old = states_new
        states = states_new - states_old

        replay_buffers = []
        for i in range(envs.num_envs):
            replay_buffers.append(deque(maxlen=4096))


        episode_start = np.zeros(envs.num_envs, dtype=bool)

        max_reward = -9999
        n_episodes = 0
        safe_left, safe_right = 30, 120    
        gamma_punish = -0.002            
        corner_patience = 20           


        still_frames   = np.zeros(num_envs, dtype=np.int32)  
        sat_still      = 30         
        prev_paddle_x  = np.zeros(num_envs, dtype=np.int16)
        corner_frames = np.zeros(num_envs, dtype=np.int32)
        brick_combo   = np.zeros(num_envs, dtype=np.int32)   
        combo_target  = 2                                    
        serve_frames   = np.zeros(num_envs, dtype=np.int32)
        prev_ram = np.array([env.unwrapped.ale.getRAM() for env in envs.envs])
        def get_param(name: str, episode: int):
            sched = SCHEDULE[name]
            for i, (ep, val) in enumerate(sched):
                if episode < ep:
                    return sched[i-1][1] if i else val
            return sched[-1][1]
        while True:
            entropy_weight = get_param("entropy_weight", n_episodes)
            lambda_still   = get_param("lambda_still",   n_episodes)
            combo_bonus    = get_param("combo_bonus",    n_episodes)
            move_bonus     = get_param("move_bonus",     n_episodes)
            beta_bonus     = get_param("beta_bonus",     n_episodes)
            noise_sigma    = get_param("noise_sigma",    n_episodes)
            
            with torch.no_grad():
                action_dist = policy_nn(states).cpu().numpy()
            curr_ram = np.array([env.unwrapped.ale.getRAM() for env in envs.envs])

            actions = []
            extra_punish = np.zeros(num_envs, dtype=np.float32)
            for i in range(num_envs):
                if curr_ram[i][0x58] & 0x02:  
                    actions.append(1)  
                    serve_frames[i] += 1
                    if serve_frames[i] > 120:
                        extra_punish[i] = -1.0  
                else:
                    serve_frames[i] = 0
                    
                    if np.allclose(action_dist[i], action_dist[i][0]):
                        actions.append(1)  
                    else:
                        actions.append(np.random.choice(all_actions, p=action_dist[i]))
            aux_states, rewards, terminations, truncations, info = envs.step(actions)

            rewards += extra_punish   
            extra_punish[:] = 0.0   
            if game_name == "ALE/Breakout-v5":
                for i in range(num_envs):
                    paddle_x = int(curr_ram[i][0x72])

                    if paddle_x == prev_paddle_x[i]:
                        still_frames[i] += 1
                        punish = -lambda_still * min(still_frames[i], sat_still) / sat_still
                        rewards[i] += punish
                    else:
                        still_frames[i] = 0 
                    if paddle_x != prev_paddle_x[i]:
                        rewards[i] += move_bonus                    
                    prev_paddle_x[i] = paddle_x
                    broken_now = 0
                    for addr in range(0x40, 0x46):
                        diff = prev_ram[i][addr] & (~curr_ram[i][addr])
                        broken_now += bin(diff).count("1")

                    if broken_now:
                        brick_combo[i] += broken_now
                        if brick_combo[i] >= combo_target:
                            rewards[i] += combo_bonus 
                            brick_combo[i] = 0

                    prev_ram[i] = curr_ram[i]                   
                            
            for i in range(num_envs):
                current_lives = info["lives"][i] if isinstance(info, dict) else info[i]["lives"]

                if game_name == "ALE/Breakout-v5" and current_lives < prev_lives[i]:
                    rewards[i] = -0.3 
                    brick_combo[i] = 0

                prev_lives[i] = current_lives
            if game_name == "ALE/Breakout-v5": 
                
                for i in range(num_envs):
                    paddle_x = curr_ram[i][0x72]

                    if safe_left <= paddle_x <= safe_right:
                        rewards[i] += beta_bonus
                        corner_frames[i] = 0              

                    else:
                        corner_frames[i] += 1
                        if corner_frames[i] > corner_patience:
                            rewards[i] += gamma_punish
                            corner_frames[i] = corner_patience   


            states_old = states_new
            if game_name == "ALE/Breakout-v5":
                states_new = torch.from_numpy(aux_states).to(device) \
                   .permute(0,3,1,2).float() / 255.0
            else:
                states_new = torch.tensor(aux_states, dtype=torch.float32) \
                            .flatten(start_dim=1).to(device) / 255


            for i in range(envs.num_envs):

                if not episode_start[i]:
                    replay_buffers[i].append({
                        "state":states[i],
                        "frame": aux_states[i],
                        "action": int(actions[i]),
                        "reward": float(rewards[i]),
                        "termination": bool(terminations[i]),
                        "truncation": bool(truncations[i]),
                    })

            states = states_new - states_old
            episode_start = np.logical_or(terminations, truncations)
            still_frames  = np.where(episode_start, 0, still_frames)
            brick_combo = np.where(episode_start, 0, brick_combo)
            prev_paddle_x = np.where(episode_start, 0, prev_paddle_x)
            corner_frames = np.where(episode_start, 0, corner_frames)


            for i in range(envs.num_envs):
                if episode_start[i]:
                    still_frames[i] = 0     
                    n_episodes += 1
                    pbar.update()
                    replay = list(replay_buffers[i])


                    j0 = -1
                    for j, r in enumerate(replay[:-1]):
                        if r['termination'] or r['truncation']:
                            j0 = j
                    j0+=1
                    replay = replay[j0:]

                    rewards = [x["reward"] for x in replay]
                    sum_rewards = sum(rewards)
                    self.writer.add_scalar('reward', sum_rewards, n_episodes)

                    propagated_rewards = []

                    running_mean = 0
                    for R in rewards[::-1]:
                        if abs(R) > .5:
                            running_mean = 0
                        running_mean = R + .99 * running_mean
                        propagated_rewards.insert(0, running_mean)

                    n_nonzero_rewards = sum([abs(x)>.5 for x in rewards])
                    replay_states = torch.stack([x["state"] for x in replay])



                    if n_nonzero_rewards > 0:
                            R_tensor = torch.as_tensor(propagated_rewards, dtype=torch.float32, device=device)

                            baseline = 0.95 * baseline + 0.05 * R_tensor.mean()
                            adv      = torch.clamp(R_tensor - baseline, -1.0, 1.0)

                            probs   = torch.clamp(policy_nn(replay_states), 1e-6, 1.0)
                            entropy = -(probs * torch.log(probs)).sum(1).mean()

                            actions_taken   = torch.tensor([step["action"] for step in replay],
                                                        dtype=torch.long, device=device)
                            log_probs_taken = torch.log(probs[torch.arange(len(replay)), actions_taken])

                            loss = -(log_probs_taken * adv).mean() - entropy_weight * entropy

                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 10.0)
                            optimizer.step()
                            if n_episodes % self.hparams["check_point"] == 0:
                                chk_dir = Path(self.output_folder) / "checkpoints"
                                chk_dir.mkdir(parents=True, exist_ok=True)
                                torch.save(
                                    {
                                        "episode": n_episodes,
                                        "model_state": policy_nn.state_dict(),
                                        "optim_state": optimizer.state_dict(),
                                        "baseline": baseline,
                                    },
                                    chk_dir / f"ep{n_episodes:06d}.pt",
                                )
                            self.writer.add_scalar("loss",     loss.item(),   n_episodes)
                            self.writer.add_scalar("entropy", entropy.item(), n_episodes)

                    if (n_episodes-1) % self.hparams["video"] == 0:
                        frames = np.stack([x["frame"] for x in replay])
                        frames = np.transpose(frames, (0, 3, 1, 2))
                        frames = np.expand_dims(frames, axis=0)
                        self.writer.add_video('gameplay', frames, n_episodes, fps=30)

                    self.writer.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Train.add_arguments(parser)

    args = parser.parse_args()
    hparams = vars(args)
    t = Train(hparams)
    t.run()
