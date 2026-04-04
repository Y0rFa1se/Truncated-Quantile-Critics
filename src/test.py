import gymnasium as gym
import hydra
import wandb
import torch
import pytorch_lightning as L
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from modules.agent import Agent
from modules.buffer import ReplayBuffer
from modules.datamodule import get_dataloader

wandb_logger = WandbLogger()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.run.project_name, name=cfg.run.run_name, config=config_dict)

    env = gym.make(cfg.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(**cfg)

    buffer = ReplayBuffer(state_dim, action_dim, max_size=cfg.training.max_buffer_size)
    dataloader = get_dataloader(buffer, batch_size=cfg.training.batch_size)

    trainer = L.Trainer(
        max_epochs=cfg.training.total_episodes,
        logger=wandb_logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    print("Collecting initial experience...")
    state, _ = env.reset()
    for _ in range(cfg.training.start_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.add(state, action, reward, next_state, terminated or truncated)
        state = next_state if not (terminated or truncated) else env.reset()[0]

    print("Starting Training...")

    state, _ = env.reset()
    for epoch in range(cfg.training.total_episodes):
        for _ in range(cfg.training.sample_per_epoch):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _ = agent(state_tensor, deterministic=False)
                action = action.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.add(state, action, reward, next_state, terminated or truncated)

            state = next_state if not (terminated or truncated) else env.reset()[0]
        trainer.fit(agent, train_dataloaders=dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()
