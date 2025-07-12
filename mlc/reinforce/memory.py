class SumTree(self,):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.size = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

        if self.size < self.capacity:
            self.size += 1
        self.write = (self.write + 1) % self.capacity

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, data_idx, self.data[data_idx]

    def total(self):
        return self.tree[0]

class PriorityQueue:
    def __init__(self,capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

class MultistepReplayBuffer:
    def __init__(self, capacity = 100000, n_step = 10, gamma=0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step or done:
            state, action, _, _, _ = self.n_step_buffer[0]
            R, s_n, d_n = self._compute_n_return()
            self.buffer.append((state, action, R, s_n, d_n))
        if done:    
                        
            while len(self.n_step_buffer) > 0:
                s,a,_,_,_ = self.n_step_buffer[0]
                R, s_n, d_n = self._compute_n_return()
                if d:
                    break
            state, action, _, next_state, _ = self.n_step_buffer[0]
            self.buffer.append((state, action, R, next_state, d_n))

    def _compute_n_return(self):
        R = 0
        for i,(_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * r
            if d: break
        d_n = self.n_step_buffer[i][4]
        s_n = self.n_step_buffer[i][3]        
        return R, s_n, d_n

    def clear(self):
        self.buffer.clear()
        self.n_step_buffer.clear()
        

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states))
    
    def __len__(self):
        return len(self.buffer)