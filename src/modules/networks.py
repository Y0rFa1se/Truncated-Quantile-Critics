import torch
import torch.nn as nn
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.bias, -3e-3, 3e-3)

    def _dist(self, state):
        x = self.network(state)

        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        log_std = log_std.exp()

        return mu, log_std

    def forward(self, state, deterministic=False, with_log_prob=True):
        mu, std = self._dist(state)

        if deterministic:
            return torch.tanh(mu), None

        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        if with_log_prob:
            log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob


class QuantileEnsembleNetwork(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, critic_num=5, quantile_num=25
    ):
        super(QuantileEnsembleNetwork, self).__init__()
        self.critic_num = critic_num
        self.quantile_num = quantile_num

        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, quantile_num),
                )
                for _ in range(critic_num)
            ]
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        quantiles = torch.stack([critic(x) for critic in self.critics], dim=0)

        return quantiles
