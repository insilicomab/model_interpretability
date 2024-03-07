import json
from pathlib import Path

import captum
import hydra
import numpy as np
import torch
from captum.attr import visualization as viz
from omegaconf import DictConfig
from tqdm import tqdm

from attribution import SaliencyAttribution
from common.functions import visualize
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

    # calculate attribution by Saliency
    saliency_attr = SaliencyAttribution(model)

    with torch.no_grad():
        for input_img, original_img, file_path in tqdm(dataloader):
            input_img.requires_grad = True

            attribution_img = saliency_attr.attribute(
                inputs=input_img,
                target=cfg.target,
                abs=cfg.saliency.abs,
                additional_forward_args=cfg.saliency.additional_forward_args,
            )

            # save a figure
            visualize(
                cfg,
                attribution_img,
                original_img,
                attribution_name="Saliency",
                target_class=int_to_label[cfg.target],
                file_path=file_path[0],
            )


if __name__ == "__main__":
    main()
