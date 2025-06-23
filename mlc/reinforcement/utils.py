import numpy as np
import cv2

class Memory:
    """Replay memory
       
       METHODS
            add    -- Add transition to memory.
            sample -- Sample minibatch from memory.
            reset  -- Reset memory index.
    """
    def __init__(self, state_dims, action_dims, size=1000000):
        """Creates a new replay memory.
        
           Memory(state_dims, action_dims) creates a new replay memory for storing
           transitions with `state_dims` observation dimensions and `action_dims`
           action dimensions. It can store 1000000 transitions.
           
           Memory(state_dims, action_dims, size) additionally specifies how many
           transitions can be stored.
        """

        self.s = np.ndarray([size, *state_dims])
        self.a = np.ndarray([size, *action_dims])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, *state_dims])
        self.terminal = np.ndarray([size, 1])
        self.v = np.ndarray([size, 1])
        self.logp = np.ndarray([size, 1])
        self.adv = np.ndarray([size, 1])
        self.rtg = np.ndarray([size, 1])
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
        
        self.i = (self.i + 1) % self.size
        if self.n < self.size:
            self.n += 1
    
    def sample(self, size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.terminal[idx]
        
    def reset(self):
        """Reset memory."""

        self.i = 0
        self.n = 0

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