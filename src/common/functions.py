from pathlib import Path

import numpy as np
import torch
from captum.attr import visualization as viz
from omegaconf import DictConfig


def visualize(
    config: DictConfig,
    attribution_img: np.ndarray,
    original_img: torch.Tensor,
    attribution_name: str,
    target_class: str,
    file_path: str,
) -> None:
    if config.vis_img.enable:
        figure, _ = viz.visualize_image_attr(
            attribution_img,
            np.transpose(original_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            method=config.vis_img.method,
            sign=config.vis_img.sign,
            plt_fig_axis=config.vis_img.plt_fig_axis,
            outlier_perc=config.vis_img.outlier_perc,
            cmap=config.vis_img.cmap,
            alpha_overlay=config.vis_img.alpha_overlay,
            show_colorbar=config.vis_img.show_colorbar,
            title=f"{attribution_name} for {target_class}: {file_path}",
            fig_size=config.vis_img.fig_size,
            use_pyplot=config.vis_img.use_pyplot,
        )

        figure.savefig(str(Path(config.output_dir) / f"{attribution_name}_{file_path}"))
    # save multiple figures
    if config.vis_img_multi.enable:
        figure_m, _ = viz.visualize_image_attr_multiple(
            attribution_img,
            np.transpose(original_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            methods=config.vis_img_multi.methods,
            signs=config.vis_img_multi.signs,
            outlier_perc=config.vis_img_multi.outlier_perc,
            cmap=config.vis_img_multi.cmap,
            alpha_overlay=config.vis_img_multi.alpha_overlay,
            show_colorbar=config.vis_img_multi.show_colorbar,
            titles=[
                f"{attribution_name} for {target_class}: {file_path}",
                "original",
            ],
            fig_size=config.vis_img_multi.fig_size,
            use_pyplot=config.vis_img_multi.use_pyplot,
        )

        figure_m.savefig(
            str(Path(config.output_dir) / f"{attribution_name}_multi_{file_path}")
        )
