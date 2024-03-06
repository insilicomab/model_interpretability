import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.dataset import ImageDataset
from dataset.transformation import Transforms


def get_image_dataloader(config: DictConfig) -> DataLoader:
    # dataset
    image_dataset = ImageDataset(
        root=config.root, transform=Transforms(image_size=config.image_size)
    )

    # dataloader
    dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    return dataloader
