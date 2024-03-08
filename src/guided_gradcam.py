import json

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from attribution import GuidedGradcamAttribution
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

    # layer <=== only layer needs to be defined here !!!
    layer = model.net.norm_pre

    # get dataloader
    dataloader = get_image_dataloader(cfg)

    # calculate attribution by Guided Grad-CAM
    guided_gradcam_attr = GuidedGradcamAttribution(model, layer)

    with torch.no_grad():
        for input_img, original_img, file_path in tqdm(dataloader):
            attribution_img = guided_gradcam_attr.attribute(
                inputs=input_img,
                target=cfg.target,
                additional_forward_args=cfg.ggc.additional_forward_args,
                interpolate_mode=cfg.ggc.interpolate_mode,
                attribute_to_layer_input=cfg.ggc.attribute_to_layer_input,
            )

            # save a figure
            visualize(
                cfg,
                attribution_img,
                original_img,
                attribution_name="Guided_GradCAM",
                target_class=int_to_label[cfg.target],
                file_path=file_path[0],
            )


if __name__ == "__main__":
    main()
