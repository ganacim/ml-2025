"""ELE2761 Deep Reinforcement Learning networks

CLASSES
    V            -- State-value network
    DQ           -- State-action value network with discrete actions
    CQ           -- State-action value network with continuous actions
    Mu           -- Deterministic policy network
    Pi           -- Stochastic policy network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ttf(x):
    """Transfer float32 tensor to device."""
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, device=device, dtype=torch.float32)
def tti(x):
    """Transfer int64 tensor to device."""
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, device=device, dtype=torch.int64)

class MLP(nn.Module):
    """Basic multi-layer perceptron.
    
       METHODS
           forward     -- Perform inference.
           update      -- Train network.
           copyfrom    -- Copy weights.
    """
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, lr=1e-3):
        """Creates a new multi-layer perceptron network.
        
           MLP(sizes) creates an MLP with len(sizes) layers, with the given `sizes`.
           MLP(sizes, activation, output_activation, lr) additionally specifies the
           activation function, output activation function (`torch.nn.*`) and learning
           rate.
        """
        super().__init__()
        self.sizes = sizes
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.net = nn.Sequential(*layers).to(device)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.MSELoss()
        
    def forward(self, inputs, cpu=True):
        """Perform inference.
        
           MLP.forward(inputs) returns the output of the network for the given
           `inputs`.
        
           `inputs` can be either a single vector, or a matrix with a batch of
           vectors (one vector per row).
           
           EXAMPLE
               >>> mlp = MLP([3, 64, 64, 1])
               >>> mlp.forward([1, 2, 3])
               array([0.44394666], dtype=float32)
               >>> mlp.forward([[1, 2, 3], [4, 5, 6]])
               array([[0.44394666],
                      [0.9606236 ]], dtype=float32)
        """
        outputs = self.net(ttf(inputs))

        if cpu:
            outputs = outputs.cpu().detach().numpy()
        return outputs

    def update(self, inputs, targets):
        """Train network.
        
           update(inputs, targets) performs one gradient descent step
           on the network to approximate the mapping
           `inputs` -> `targets`.
           
           `inputs` and `targets` are matrices with batches of vectors
           (one vector per row).
           
           EXAMPLE
               >>> mlp = MLP([3, 64, 64, 1])
               >>> mlp.update([[1, 2, 3], [4, 5, 6]], [[1], [2]])
        """           
        outputs = MLP.forward(self, inputs, cpu=False)
        loss = self.criterion(outputs, ttf(targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def copyfrom(self, other):
        """Copies weights from a different instance of the same network.
        
           EXAMPLE
               >>> mlp1 = MLP([3, 64, 64, 1])
               >>> mlp2 = MLP([3, 64, 64, 1])
               >>> mlp1.copyfrom(mlp2)
        """  
        state_dict = self.state_dict()
        other_state_dict = other.state_dict()
        for key in state_dict:
            state_dict[key] = other_state_dict[key]
        self.load_state_dict(state_dict)
        
class V(MLP):
    """State-value network."""
    def __init__(self, state_dims, hiddens=[64, 64], lr=1e-3):
        """Creates a new state-value network.
        
           V(state_dims) creates a network for states with `state_dims` dimensions.
           V(state_dims, hiddens, lr) additionally specifies sizes of the hidden
           layers in `hiddens`, as well as the learning rate `lr`.
        """
        super().__init__([state_dims] + list(hiddens) + [1], lr=lr)

class DQ(MLP):
    """State-action value network with discrete actions."""
    def __init__(self, state_dims, actions, hiddens=[64, 64], lr=1e-3):
        """Creates a new state-action value network with discrete actions.
        
           DQ(state_dims, actions) creates a network for states with
           `state_dims` dimensions and `actions` actions.
           DQ(state_dims, actions, hiddens, lr) additionally specifies sizes of
           the hidden layers in `hiddens`, as well as the learning rate `lr`.
        """
        super().__init__([state_dims] + list(hiddens) + [actions], lr=lr)

    def forward(self, s, a=None, cpu=True):
        """Perform inference.
        
           DQ.forward(s) returns the values of all actions at state `s`.
           DQ.forward(s, a) returns the value of a specifc action `a` at state `s`.
           
           `s` can be either a single vector, or a matrix with a batch of vectors.
           In the latter case, `a`, when given, must also be a batch of vectors.
        
           EXAMPLE
               >>> dq = DQ(3, 3)
               >>> dq.forward([1, 2, 3])
               array([0.16732767, 0.33436352, 0.19102027], dtype=float32)
               >>> dq.forward([[1, 2, 3], [4, 5, 6]], [[1], [2]])
               array([[0.33436352],
                      [0.5879653 ]], dtype=float32)
        """
        if a is None:
            q = super().forward(s, cpu=False)
        else:
            q = super().forward(s, cpu=False).gather(1, tti(a))

        if cpu:
            q = q.cpu().detach().numpy()
        return q        
        
    def update(self, s, a, targets):
        """Train network.
        
           update(s, a, targets) performs one gradient descent step
           on the network to approximate the mapping
           (`s`, `a`) -> `targets`.
           
           `s`, `a`, and `targets` are matrices with batches of vectors
           (one vector per row).
           
           EXAMPLE
               >>> dq = DQ(3, 3)
               >>> dq.update([[1, 2, 3], [4, 5, 6]], [[1], [2]], [[0.5], [1.5]])
        """           
        outputs = self.forward(s, a, cpu=False)
        loss = self.criterion(outputs, ttf(targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class CQ(MLP):
    """State-action value network with continuous actions."""
    def __init__(self, state_dims, action_dims, hiddens=[64, 64], lr=1e-3):
        """Creates a new state-action value network with continuous actions.
        
           CQ(state_dims, action_dims) creates a network for states with
           `state_dims` dimensions and `action_dims` action dimensions.
           CQ(state_dims, action_dims, hiddens, lr) additionally specifies sizes of
           the hidden layers in `hiddens`, as well as the learning rate `lr`.
        """
        super().__init__([state_dims + action_dims] + list(hiddens) + [1], lr=lr)

    def forward(self, s, a, cpu=True):
        """Perform inference.
        
           CQ.forward(s, a) returns the value of action `a` at state `s`.
           
           `s` and `a` can be either single vectors, or matrices with batches of
           vectors.
        
           EXAMPLE
               >>> cq = CQ(3, 1)
               >>> cq.forward([1, 2, 3], [0.5])
               array([0.42496216], dtype=float32)
               >>> cq.forward([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]])
               array([[0.42496222],
                      [0.8978188 ]], dtype=float32)
        """
        if np.asarray(s).ndim == 1:
            q = super().forward(torch.concatenate((ttf(s), ttf(a))), cpu=False)
        else:
            q = super().forward(torch.concatenate((ttf(s), ttf(a)), axis=1), cpu=False)
       
        if cpu:
            q = q.cpu().detach().numpy()
        return q
        
    def update(self, s, a, targets):
        """Train network.
        
           update(s, a, targets) performs one gradient descent step
           on the network to approximate the mapping
           (`s`, `a`) -> `targets`.
           
           `s`, `a`, and `targets` are matrices with batches of vectors
           (one vector per row).
           
           EXAMPLE
               >>> cq = DQ(3, 1)
               >>> cq.update([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]], [[0.5], [1.5]])
        """           
        super().update(np.concatenate((s, a), axis=1), targets)

class Mu(MLP):
    """Deterministic policy network."""
    def __init__(self, state_dims, action_dims, hiddens=[64, 64], lr=1e-4):
        """Creates a new deterministic policy network.
        
           Mu(state_dims, action_dims) creates a network for states with
           `state_dims` dimensions and `action_dims` action dimensions.
           Mu(state_dims, action_dims, hiddens, lr) additionally specifies sizes of
           the hidden layers in `hiddens`, as well as the learning rate `lr`.
           
           The action is in the range [-1, 1].
        """
        super().__init__([state_dims] + list(hiddens) + [action_dims], output_activation=nn.Tanh, lr=lr)
        
    def update(self, s, q_net):
        """Train network.
        
           update(s, q_net) performs one gradient descent step on the network
           to increase the value of q_net(s, Mu(s)).
           
           `s` is a matrix with a batch of vectors (one vector per row).
           
           EXAMPLE
               >>> cq = CQ(3, 1)
               >>> mu = Mu(3, 1)
               >>> mu.update([[1, 2, 3], [4, 5, 6]], cq)
        """           
        loss = -q_net.forward(s, self.forward(s, cpu=False), cpu=False).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Pi(MLP):
    """Stochastic policy network."""
    def __init__(self, state_dims, action_dims, hiddens=[64, 64], lr=3e-4):
        """Creates a new stochastic policy network.
        
           Pi(state_dims, action_dims) creates a network for states with
           `state_dims` dimensions and `action_dims` action dimensions.
           Pi(state_dims, action_dims, hiddens, lr) additionally specifies sizes of
           the hidden layers in `hiddens`, as well as the learning rate `lr`.
           
           The action mean is in the range [-1, 1]. The initial log standard
           deviation of the action distribution is -0.5.
        """
        super().__init__([state_dims] + list(hiddens) + [action_dims], output_activation=nn.Tanh, lr=lr)
        log_std = -0.5 * np.ones(action_dims, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std).to(device))
        
        # Reinitialize optimizer because we added a parameter
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

    def forward(self, s, a_prev=None, cpu=True):
        """Perform inference.
        
           Pi.forward(s) returns an action sampled from the policy distribution at
           state `s`, along with the log-probability of that action.
           Pi.forward(s, a_prev) returns an action sampled from the policy
           distribution at state `s`, along with the log-probability of taking
           action `a_prev` at that same state.
           
           `s` can be either a single vector, or a matrix with a batch of
           vectors. In the latter case, `a_prev`, when given, must also be a batch
           of vectors.
        
           EXAMPLE
               >>> pi = Pi(3, 1)
               >>> pi.forward([1, 2, 3])
               (array([0.3648948], dtype=float32), array([-0.48902923], dtype=float32))
               >>> pi.forward([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]])
               (array([[-0.19757865],
                       [ 0.336668  ]], dtype=float32),
                array([[-0.45469385],
                       [-0.56612885]], dtype=float32))
        """
        mu = super().forward(s, cpu=False)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        a = dist.sample()
        if a_prev is None:
            # Return log probability of taking sampled action
            logp = dist.log_prob(a)
        else:
            # Return log probability of taking action a_prev
            logp = dist.log_prob(ttf(a_prev))
            
        if cpu:
            a = a.cpu().detach().numpy()
            logp = logp.cpu().detach().numpy()
        return a, logp

    def update(self, s, a_prev, logp_a, adv, clip_ratio=0.2):
        """Train network.
        
           update(s, a_prev, logp_a, adv) performs one gradient descent step
           on the network to minimize the clipped PPO objective function,
           using the advantages `adv` of taking actions `a_prev` at
           states `s`, with probabilities `a_prev`.
           update(s, a_prev, logp_a, adv, clip_ratio) additionally specifies
           the `clip_ratio` that constrains the size of the update.
           
           `s`, `a_prev`, `logp_a` and `adv` are matrices with batches of
           vectors (one vector per row).
           
           EXAMPLE
               >>> pi = Pi(3, 1)
               >>> pi.update([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]], [[-0.45], [-0.56]], [[1.12], [-0.23]])
        """           
        _, logp = self.forward(s, a_prev, cpu=False)
        ratio = torch.exp(logp - ttf(logp_a))
        adv_t = ttf(adv)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_t
        loss = -(torch.min(ratio * adv_t, clip_adv)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
