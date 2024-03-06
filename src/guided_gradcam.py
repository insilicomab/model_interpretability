import json

import hydra
import torch
from omegaconf import DictConfig

from dataset import get_image_dataloader
from model import get_model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # read label_map and generate int_to_label
    with open(cfg.label_map_path, "r") as f:
        label_map = json.load(f)
    int_to_label = {v: k for k, v in label_map.items()}

    print(f"target class: {int_to_label[cfg.target]}")

    # load model
    model = get_model(cfg)

    # get dataloader
    dataloader = get_image_dataloader(cfg)

    data_iter = iter(dataloader)
    a, b, c = next(data_iter)
    print(f"input: {type(a)}")
    print(f"original: {type(b)}")
    print(f"file_path: {c}")


if __name__ == "__main__":
    main()
