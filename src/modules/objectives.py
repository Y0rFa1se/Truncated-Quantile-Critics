import torch


def _get_critic_loss(quantiles, target_quantiles, kappa=1.0):
    N_critics, batch_size, M = quantiles.shape
    N_truncated = target_quantiles.shape[-1]

    theta = quantiles.transpose(0, 1).reshape(batch_size, -1, 1)

    target_theta = target_quantiles.view(batch_size, 1, N_truncated)

    tau = torch.arange(0.5, M, 1.0, device=quantiles.device) / M
    tau = tau.view(1, 1, M).repeat(1, N_critics, 1).view(1, -1, 1)

    delta = target_theta - theta
    delta_abs = delta.abs()

    huber_loss = torch.where(
        delta_abs <= kappa, 0.5 * delta.pow(2), kappa * (delta_abs - 0.5 * kappa)
    )

    quantile_weight = torch.abs(tau - (delta < 0).float())

    loss = (quantile_weight * huber_loss).mean()

    return loss


def _get_actor_loss(log_prob_n, quantiles, alpha):
    q_values = quantiles.mean(dim=(0, 2)).unsqueeze(-1)
    actor_loss = (alpha * log_prob_n - q_values).mean()

    return actor_loss


def _get_log_alpha_loss(log_alpha, log_prob, target_entropy):
    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

    return alpha_loss


def get_critic_loss(agent, batch):
    state, action, reward, next_state, done = batch
    quantiles = agent.critic(state, action)

    with torch.no_grad():
        action_n, log_prob_n = agent(next_state)
        target_quantiles = agent.target_critic(next_state, action_n)
        target_quantiles -= agent.alpha * log_prob_n.unsqueeze(0)

        r = reward.view(1, -1, 1)
        d = done.view(1, -1, 1)

        target_quantiles = r + agent.gamma * (1 - d) * target_quantiles

        target_quantiles = target_quantiles.transpose(0, 1).flatten(1)
        target_quantiles, _ = torch.sort(target_quantiles, dim=1)

        target_quantiles = target_quantiles[:, : agent.total_quantiles - agent.drop_k]

    return _get_critic_loss(quantiles, target_quantiles, agent.kappa)


def get_actor_loss(agent, batch):
    state, action, reward, next_state, done = batch
    action_n, log_prob_n = agent(state)
    quantiles = agent.critic(state, action_n)

    return _get_actor_loss(log_prob_n, quantiles, agent.alpha)


def get_log_alpha_loss(agent, batch):
    state, action, reward, next_state, done = batch
    action_n, log_prob_n = agent(state)

    return _get_log_alpha_loss(agent.log_alpha, log_prob_n, agent.target_entropy)
