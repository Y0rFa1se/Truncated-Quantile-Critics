import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        project=cfg.run.project_name, name=cfg.run.run_name, config=config_dict
    )
    
    wandb.finish()


if __name__ == "__main__":
    main()
