import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayBuffer
from nn_model import QNetwork


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, agent_hyperparams):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): the number of actions
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.params = agent_hyperparams
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(
            self.params["device"]
        )
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(
            self.params["device"]
        )
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.params["LR"]
        )

        self.memory = ReplayBuffer(
            action_size,
            self.params["BUFFER_SIZE"],
            self.params["BATCH_SIZE"],
            seed,
            self.params["device"],
        )
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.params["UPDATE_EVERY"]
        if self.t_step == 0:
            if len(self.memory) > self.params["BATCH_SIZE"]:
                experiences = self.memory.sample()
                self.learn(experiences, self.params["GAMMA"])

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.params["device"])
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params["TAU"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
