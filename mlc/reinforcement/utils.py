import numpy as np
import cv2
import torch
from gymnasium.wrappers import RecordVideo
import os

def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
def record_model(env, model, video_folder, name_prefix = ""):
    if not os.path.isdir(video_folder):
        os.makedirs(video_folder)
    env_name = env.unwrapped.spec.id
    env = RecordVideo(env, video_folder=video_folder, name_prefix = env_name + "_" + name_prefix )

    obs, _ = env.reset()
    done = False
    total_reward = 0

    device = next(model.parameters()).device

    while not done:
        # --- Replace this with your model's action ---
        # action = model.predict(obs)  # if model-based
        if model == None:
            action = env.action_space.sample()  # random action for demo
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                action = model(obs_tensor).detach().cpu().squeeze().numpy()  # Get action from actor network
                #action = action + np.random.normal(0, 0.1)  # Noise = Normal(0,sigma)?
                action = np.clip(action * 1.1, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode finished. Total reward: {total_reward}")
    env.close()

class Memory:
    """Replay memory
       
       METHODS
            add    -- Add transition to memory.
            sample -- Sample minibatch from memory.
            reset  -- Reset memory index.
    """
    def __init__(self, state_dims, action_dims, size=1000000, alpha = 0.6):
        """Creates a new replay memory.
        
           Memory(state_dims, action_dims) creates a new replay memory for storing
           transitions with `state_dims` observation dimensions and `action_dims`
           action dimensions. It can store 1000000 transitions.
           
           Memory(state_dims, action_dims, size) additionally specifies how many
           transitions can be stored.
        """
        self.s = np.ndarray([size, state_dims])
        self.a = np.ndarray([size, action_dims])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, state_dims])
        self.terminal = np.ndarray([size, 1])
        self.v = np.ndarray([size, 1])
        self.logp = np.ndarray([size, 1])
        self.adv = np.ndarray([size, 1])
        self.rtg = np.ndarray([size, 1])
        self.priority = np.zeros([size])
        self.max_prio = 1.0
        self.alpha = alpha

        self.i = 0
        self.n = 0
        self.size = size
    
    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""

        return self.n
    
    def add(self, s, a, r, sp, terminal, v=0, logp=0):
        """Adds a transition to the replay memory.
        
           Memory.add(s, a, r, sp, terminal) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `terminal`
           specifies whether the episode finished at terminal absorbing
           state `sp`.

           Memory.add(s, a, r, sp, terminal, v, logp) additionally records
           the value of state s and log-probability of taking action a.
        """

        self.s[self.i, :] = s
        self.a[self.i, :] = a
        self.r[self.i, :] = r
        self.sp[self.i, :] = sp
        self.terminal[self.i, :] = terminal
        self.v[self.i, :] = v
        self.logp[self.i, :] = logp
        self.priority[self.i] = self.max_prio
        
        self.i = (self.i + 1) % self.size
        if self.n < self.size:
            self.n += 1
    
    def sample(self, batch_size, beta = 0.4):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        #idx = np.random.randint(0, self.n, batch_size)
        if self.n == self.size:
            prios = self.priority
        else:
            prios = self.priority[:self.n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(self.n, batch_size, p=probs)

        weights = (self.n * probs[idx]) ** (-beta)
        weights /= weights.max()

        return [self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.terminal[idx]], weights, idx
        
    def reset(self):
        """Reset memory."""

        self.i = 0
        self.n = 0

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priority[idx] = abs(err) + 1e-8
            self.max_prio = max(self.priority[idx], self.max_prio)


class PreprocessFrame():
    """
    Preprocesses an input frame:
    - Converts to grayscale
    - Resizes image
    - Returns as float32 normalized between [0, 1]

    Args:
        frame: (H, W, 3) RGB image from the env
        resize_shape: output shape (default 84x84)

    Returns:
        (resize_shape[0], resize_shape[1]) grayscale normalized float32 frame
    """
    def __init__(self, use_grayscale = False, resize_shape = None):
        self.gs = use_grayscale
        self.reshape = resize_shape

    def __call__(self, frame):
        if self.gs:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Grayscale
        if self.reshape:
            frame = cv2.resize(frame, self.reshape, interpolation=cv2.INTER_AREA)  # Resize
        frame = frame.astype(np.float32) / 255.0  # Normalize
        return np.atleast_3d(frame)