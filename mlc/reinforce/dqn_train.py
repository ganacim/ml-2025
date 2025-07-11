import argparse
import random
import re
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import os
from torch import nn 
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from mlc.command.base import Base
from mlc.reinforce.car_nets import ModeloDQN as Modelo
from mlc.util.resources import get_time_as_str


class TrainDQN(Base):

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
        self.gamma = hparams["gamma"]
        # MODIFICADO: DQN funciona com ações discretas
        self.mode = "discrete" 
        if hparams["mode"] == "continuous":
            print("AVISO: DQN clássico não suporta ações contínuas. Usando modo 'discrete'.")
        
        # latest_link_path = f"{agent_folder}/latest"
        # # Remove o link antigo, se existir (lexists é seguro para links quebrados)
        # if os.path.lexists(latest_link_path):
        #     os.remove(latest_link_path)
        # # Cria um novo link simbólico apontando para a pasta da execução atual
        # os.symlink(run_name, latest_link_path)
        # print(f"Link 'latest' criado, apontando para: {self.output_folder}")
            
        self.num_stack = hparams["num_stack"]
        self.lr_decay = hparams["lr_decay"]
        self.learning_rate = hparams["learning_rate"]
        if hparams["name"]:
            self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/dqn_agent/{hparams['name']}"
        else:
            self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/dqn_agent/{get_time_as_str()}"
        os.makedirs(f"{self.output_folder}/checkpoints", exist_ok=True)
        self.writer = SummaryWriter(self.output_folder + "/tensorboard")
        gym.register_envs(ale_py)
        
        # ADICIONADO: Hiperparâmetros específicos do DQN
        self.batch_size = hparams["batch_size"]
        self.epsilon_start = hparams["epsilon_start"]
        self.epsilon_end = hparams["epsilon_end"]
        self.epsilon_decay = hparams["epsilon_decay"]
        self.target_update_freq = hparams["target_update"]

    @classmethod
    def name(cls):
        return "traindqn_car"

    @staticmethod
    def add_arguments(parser):
        def _parse_device_arg(arg_value):
            pattern = re.compile(r"(cpu|cuda|cuda:\d+)")
            if not pattern.match(arg_value):
                raise argparse.ArgumentTypeError("invalid value")
            return arg_value

        parser.add_argument("-s", "--seed", type=int, default=42)
        parser.add_argument("-e", "--max_episodes", type=int, default=2000)

        parser.add_argument("-g", "--game", default="CarRacing-v3")
        parser.add_argument("--num_envs", default=1, type=int)
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="learning rate for the optimizer")
        parser.add_argument("-c", "--check-point", type=int, default=8000, help="check point every n steps")
        parser.add_argument("--resume-from", type=str, default=None, help="path to checkpoint to resume training from")
        parser.add_argument("-v", "--video", type=int, default=15, help="create a video every n episodes") #20
        parser.add_argument("-n", "--name", type=str, default=None, help="name this run")
        parser.add_argument("--gamma", type=float, default=0.95, help="discount factor for rewards")
        # O modo é fixado para discreto, mas o argumento é mantido para compatibilidade
        parser.add_argument("--mode", type=str, default="discrete", choices=["discrete", "continuous"], help="mode of the agent")
        parser.add_argument("--lr_decay", default=False, action="store_true", help="enable learning rate decay")
        parser.add_argument("--num_stack", type=int, default=4, help="number of frames to stack for the agent input")
        parser.add_argument("--validation", type=str, default=None, help="path to validation script (not used in training)")
        
        # Argumentos para DQN
        parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size for training")
        parser.add_argument("--buffer-size", type=int, default=10000, help="size of the replay buffer")
        parser.add_argument("--epsilon-start", type=float, default=1, help="starting value of epsilon")
        parser.add_argument("--epsilon-end", type=float, default=0.05, help="final value of epsilon")
        parser.add_argument("--epsilon-decay", type=float, default=30000, help="epsilon decay rate") # quanto menor, maior a velocidade de decaimento
        parser.add_argument("--target-update", type=int, default=5, help="frequency of target network updates")
        parser.add_argument("--learning-starts", type=int, default=10000, help="number of steps before starting training")

    # Função para selecionar ação com epsilon-greedy
    def select_action(self, state, policy_net, n_actions, steps_done):
        #steps_done = steps_done-self.learning_rate
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * steps_done / self.epsilon_decay)
        # if steps_done> 20000:
        #     epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * steps_done / 4*self.epsilon_decay)
        actions = []
        # Para cada ambiente no vetor

        if random.random() > epsilon and steps_done > self.batch_size:
            with torch.no_grad():
                # Pega o valor Q para o estado do ambiente i
                policy_net.eval() # Modo de avaliação
                state = state.to(self.device) # Move o estado para o dispositivo correto
                q_values = policy_net(state)
                # Escolhe a ação com maior valor Q
                actions = q_values.max(1)[1] # Índices das ações com maior Q-value 
                actions = [a.item() for a in actions]
                policy_net.train() # Volta para o modo de treinamento
        else:
            actions = [random.randrange(n_actions) for _ in range(len(state))]
        return actions, epsilon
    
    # Função para otimizar o modelo (fazer o update do DQN)
    def optimize_model(self, policy_net, target_net, optimizer, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None # Não treina se o buffer não tiver amostras suficientes

        # Amostra um batch do replay buffer
        transitions = random.sample(list(replay_buffer), self.batch_size)
        
        # Converte o batch de transições para tensores
        batch = list(zip(*transitions))
        state_batch = torch.stack(batch[0]).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack(batch[3]).to(self.device)
        termination_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        
        # 1. Calcula Q(s_t, a) - O modelo calcula Q(s_t), e então selecionamos as colunas das ações tomadas
        q_values = policy_net(state_batch).gather(1, action_batch)

        # 2. Calcula V(s_{t+1}) para todos os próximos estados.       
        # Usa a target_net para maior estabilidade.
        with torch.no_grad():
            next_q_values = target_net(next_state_batch).max(1)[0]
            # O valor do próximo estado é 0 se o episódio terminou.
            next_q_values[termination_batch.bool()] = 0.0

        cond = torch.tensor([a.item()==0 for a in action_batch])
        incentivo = torch.where(cond, -0.1,0.0).to(self.device) # penaliza ficar parado
        reward_batch += incentivo
        # 3. Calcula o valor Q esperado (alvo)
        # target = r + gamma * max_a' Q_target(s', a')
        target_q_values = reward_batch + (self.gamma * next_q_values)

        # 4. Calcula o loss (MSE)
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, target_q_values.unsqueeze(1))

        # 5. Otimiza o modelo
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # Opcional: Gradiente clipping
        optimizer.step()
        
        return loss.item()

    def run(self):
        num_envs = self.hparams["num_envs"]
        device = self.device

        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.wrappers.FrameStackObservation(
                    gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, 
                             domain_randomize=False, continuous=False), # DQN é para ações discretas
                    stack_size=self.num_stack
                )
                for _ in range(num_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )

        n_actions = envs.single_action_space.n

        # Criação das duas redes: policy e target
        # A classe Modelo deve retornar Q-values (sem softmax no final)
        policy_net = Modelo( 
                            dim_hidden=64,
                            init_ch=3*self.num_stack,
                            dim_out=n_actions # ADICIONADO: Passa o número de ações para a rede
                           ).to(device)
        target_net = Modelo( 
                            dim_hidden=64,
                            init_ch=3*self.num_stack,
                            dim_out=n_actions
                           ).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval() # Rede alvo fica em modo de avaliação

        learning_rate = torch.tensor(self.hparams["learning_rate"], dtype=torch.float32)
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        
        replay_buffer = deque(maxlen=self.hparams["buffer_size"])
        
        episodes_done = 0
        start_step = 1

        if self.hparams["resume_from"] and os.path.exists(self.hparams["resume_from"]):
            print(f"Retomando treinamento do checkpoint: {self.hparams['resume_from']}")
            checkpoint = torch.load(self.hparams["resume_from"], map_location=device)

            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Sincroniza a target_net com a policy_net carregada
            target_net.load_state_dict(policy_net.state_dict())
            
            episodes_done = checkpoint['episode']
            start_step = checkpoint['step'] + 1
            
            print(f"Checkpoint carregado. Começando do episódio {episodes_done}, passo {start_step}.")
       
        
        def process_obs(obs):
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32) / 255.0
            B, S, H, W, C = obs_tensor.shape
            return obs_tensor.permute(0, 1, 4, 2, 3).reshape(B, S * C, H, W)

        # Loop de treinamento principal
        print("Iniciando o treinamento...")
        pbar = tqdm(total=self.hparams["max_episodes"],initial=episodes_done, desc="Episódios concluídos")
        
        states, _ = envs.reset(options={"randomize": False})
        states = process_obs(states).to(device)
        print(f"Estado inicial: {states.shape}, Ações possíveis: {n_actions}")
        episode_rewards = [0.0 for _ in range(num_envs)]
        episode_frames = [[] for _ in range(num_envs)]
        
        # Loop baseado em passos (steps) 
        for step in range(start_step, int(1e7)): # Loop "infinito"
            # Seleciona a ação usando epsilon-greedy
            actions, epsilon = self.select_action(states, policy_net, n_actions, step)
            self.writer.add_scalar("hyperparameters/epsilon", epsilon, step)
            
            # Executa a ação no ambiente
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            for i in range(num_envs):
                # O shape de next_obs é (num_envs, stack, H, W, C). Pegamos o último frame da pilha.
                episode_frames[i].append(next_obs[i][-1])
            
            next_states = process_obs(next_obs).to(device)
            dones = np.logical_or(terminations, truncations)

            # Armazena as transições no replay buffer
            for i in range(num_envs):
                # Armazena uma transição para cada ambiente
                replay_buffer.append((
                    states[i].cpu(), 
                    actions[i], 
                    rewards[i], 
                    next_states[i].cpu(), 
                    terminations[i]
                ))
                episode_rewards[i] += rewards[i]

                # Se um episódio terminou
                if dones[i]:
                    episodes_done += 1
                    pbar.update(1)
                    self.writer.add_scalar("reward", episode_rewards[i], episodes_done)
                    episode_rewards[i] = 0.0 # Reseta a recompensa do episódio
                                
                    if episodes_done>0 and episodes_done % self.hparams["video"] == 0:
                        # Converte a lista de frames (T, H, W, C) para um tensor (N, T, C, H, W)
                        video_array = np.array(episode_frames[i], dtype=np.uint8).transpose(0, 3, 1, 2)
                        vid_tensor = torch.from_numpy(video_array).unsqueeze(0)
                        
                        self.writer.add_video("gameplay", vid_tensor, global_step=episodes_done, fps=30)
                    self.writer.flush()
                    episode_frames[i] = [] # Reseta os frames do episódio
                    
                    if episodes_done >= self.hparams["max_episodes"]:
                        break
            
            if episodes_done >= self.hparams["max_episodes"]:
                break

            states = next_states
                                 
            if step == self.hparams["learning_starts"]:
                print(f"Passo {step}, Episódios concluídos: {episodes_done}, Epsilon: {epsilon:.4f}")
            # Treina a rede
            if step > self.hparams["learning_starts"]:
                loss = self.optimize_model(policy_net, target_net, optimizer, replay_buffer)
                if loss is not None:
                    self.writer.add_scalar("loss", loss, step)

            # Atualiza a target network
            if episodes_done % self.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # Checkpoint do modelo
            if step>0 and step % self.hparams["check_point"] == 0: # Ajuste a frequência de checkpoint
                checkpoint_path = f'{self.output_folder}/checkpoints/{step:08d}.pt'
                torch.save({
                    'episode': episodes_done,
                    'step': step,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

        pbar.close()
        print("Treinamento concluído.")
        torch.save(policy_net.state_dict(), f'{self.output_folder}/final_model.pt')
        self.writer.close()
        envs.close()

    def validation(path_dict):
        pass# Carrega o modelo
        # path_dict = 
        # print(f"Carregando modelo de {path_dict['model']}")
        # model = ModeloDQN()
        # model.load_state_dict(torch.load(path_dict["model"]))
        # model.eval()

        # # Inicializa o ambiente
        # env = gym.make(path_dict["env"])
        # env = gym.vector.AsyncVectorEnv(
        #         [
        #             lambda: gym.wrappers.FrameStackObservation(
        #                 gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, 
        #                         domain_randomize=False, continuous=False), # DQN é para ações discretas
        #                 stack_size=self.num_stack
        #             )
        #             for _ in range(num_envs)
        #         ],
        #         autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        #     )

        #     n_actions = envs.single_action_space.n

        # obs, _ = env.reset()

        # with torch.no_grad():
        #     while True:
        #         # Seleciona a ação
        #         action = model.act(obs)
        #         obs, reward, done, info = env.step(action)

        #         if done:
        #             break

        # env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser) # Nome da classe corrigido

    args = parser.parse_args()
    hparams = vars(args)
    t = TrainDQN(hparams) # Nome da classe corrigido
    if hparams["validation"] and os.path.exists(hparams["validation"]):
        t.validation(hparams["validation"])
    t.run()
