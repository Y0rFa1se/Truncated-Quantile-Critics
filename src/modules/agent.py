import pytorch_lightning as L
import torch

from .networks import PolicyNetwork, QuantileEnsembleNetwork
from .objectives import get_critic_loss, get_actor_loss, get_log_alpha_loss


class Agent(L.LightningModule):
    def __init__(self, **kwargs):
        super(Agent, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.state_dim = self.hparams.network.state_dim
        self.action_dim = self.hparams.network.action_dim
        self.policy_hidden_dim = self.hparams.network.policy_hidden_dim
        self.critic_hidden_dim = self.hparams.network.critic_hidden_dim
        self.critic_num = self.hparams.network.critic_num
        self.quantile_num = self.hparams.network.quantile_num
        self.log_alpha = torch.nn.Parameter(torch.tensor(float(self.hparams.network.log_alpha)))
        self.alpha = self.log_alpha.exp()
        self.gamma = self.hparams.training.gamma
        self.kappa = self.hparams.training.kappa

        self.actor = PolicyNetwork(self.state_dim, self.action_dim, self.policy_hidden_dim)
        self.quantiles = QuantileEnsembleNetwork(
            self.state_dim, self.action_dim, self.critic_hidden_dim, self.critic_num, self.quantile_num
        )
        self.target_quantiles = QuantileEnsembleNetwork(
            self.state_dim, self.action_dim, self.critic_hidden_dim, self.critic_num, self.quantile_num
        )
        self.target_quantiles.load_state_dict(self.quantiles.state_dict())

        self.total_quantiles = self.critic_num * self.quantile_num
        self.drop_k = (
            self.hparams.training.drop_k if self.hparams.training.drop_k else 0
        )

        if self.hparams.training.target_entropy:
            self.target_entropy = self.hparams.training.target_entropy
        else:
            self.target_entropy = -float(self.action_dim)

    def forward(self, state, deterministic=False, with_log_prob=True):
        return self.actor(state, deterministic, with_log_prob)

    def critic(self, state, action):
        return self.quantiles(state, action)

    def target_critic(self, state, action):
        return self.target_quantiles(state, action)

    def configure_optimizers(self):
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.training.policy_lr)
        critic_opt = torch.optim.Adam(
            self.quantiles.parameters(), lr=self.hparams.training.critic_lr
        )
        log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.hparams.training.log_alpha_lr)

        return [actor_opt, critic_opt, log_alpha_opt]

    def training_step(self, batch, batch_idx):
        actor_opt, critic_opt, log_alpha_opt = self.optimizers()

        critic_loss = get_critic_loss(self, batch)
        critic_opt.zero_grad()
        self.manual_backward(critic_loss)
        critic_opt.step()

        actor_loss = get_actor_loss(self, batch)
        actor_opt.zero_grad()
        self.manual_backward(actor_loss)
        actor_opt.step()

        log_alpha_loss = get_log_alpha_loss(self, batch)
        log_alpha_opt.zero_grad()
        self.manual_backward(log_alpha_loss)
        log_alpha_opt.step()

        self._soft_update()
        self.alpha = self.log_alpha.exp()

        self.log_dict(
            {
                "train/reward": batch[2].mean(),
                "train/critic_loss": critic_loss,
                "train/actor_loss": actor_loss,
                "train/alpha_loss": log_alpha_loss,
                "train/alpha": self.log_alpha.detach().exp(),
            },
            prog_bar=True,
            logger=True,
        )

    def _soft_update(self):
        tau = self.hparams.training.tau

        for target_param, param in zip(
            self.target_quantiles.parameters(), self.quantiles.parameters()
        ):
            target_param.data.lerp_(param.data, tau)
