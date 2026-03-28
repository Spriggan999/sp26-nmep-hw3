import wandb
import yaml

def train():
    # load base config
    with open("sweep_lr.yaml") as f:
        config = yaml.safe_load(f)

    wandb.init(project="ml@b-hw3-learningrates", config=config)

    # 🔑 override with sweep values
    if "train_config" in wandb.config:
        for k, v in wandb.config.train_config.items():
            config["TRAIN"][k] = v