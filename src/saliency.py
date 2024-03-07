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
    saliency_attr = SaliencyAttribution(model, int_to_label)

    with torch.no_grad():
        for input_img, original_img, file_path in tqdm(dataloader):
            input_img.requires_grad = True
            attribution_img, target_class, attribution_name = saliency_attr.attribute(
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
                attribution_name,
                target_class,
                file_path,
            )
            # if cfg.vis_img.enable:
            #     figure, _ = viz.visualize_image_attr(
            #         attribution_img,
            #         np.transpose(
            #             original_img.squeeze().cpu().detach().numpy(), (1, 2, 0)
            #         ),
            #         method=cfg.vis_img.method,
            #         sign=cfg.vis_img.sign,
            #         plt_fig_axis=cfg.vis_img.plt_fig_axis,
            #         outlier_perc=cfg.vis_img.outlier_perc,
            #         cmap=cfg.vis_img.cmap,
            #         alpha_overlay=cfg.vis_img.alpha_overlay,
            #         show_colorbar=cfg.vis_img.show_colorbar,
            #         title=f"{attribution_name} for {target_class}: {file_path[0]}",
            #         fig_size=cfg.vis_img.fig_size,
            #         use_pyplot=cfg.vis_img.use_pyplot,
            #     )

            #     figure.savefig(str(Path(cfg.output_dir) / f"Saliency_{file_path[0]}"))
            # # save multiple figures
            # if cfg.vis_img_multi.enable:
            #     figure_m, _ = viz.visualize_image_attr_multiple(
            #         attribution_img,
            #         np.transpose(
            #             original_img.squeeze().cpu().detach().numpy(), (1, 2, 0)
            #         ),
            #         methods=cfg.vis_img_multi.methods,
            #         signs=cfg.vis_img_multi.signs,
            #         outlier_perc=cfg.vis_img_multi.outlier_perc,
            #         cmap=cfg.vis_img_multi.cmap,
            #         alpha_overlay=cfg.vis_img_multi.alpha_overlay,
            #         show_colorbar=cfg.vis_img_multi.show_colorbar,
            #         titles=[
            #             "original",
            #             f"{attribution_name} for {target_class}: {file_path[0]}",
            #         ],
            #         fig_size=cfg.vis_img_multi.fig_size,
            #         use_pyplot=cfg.vis_img_multi.use_pyplot,
            #     )

            #     figure_m.savefig(
            #         str(Path(cfg.output_dir) / f"Saliency_multi_{file_path[0]}")
            #     )


if __name__ == "__main__":
    main()
