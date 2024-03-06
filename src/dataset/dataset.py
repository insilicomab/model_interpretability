import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from dataset.transformation import Transforms


class ImageDataset(Dataset):
    def __init__(self, root: str, transform: Compose) -> None:
        self.root = root
        self.image_path_list = [f for f in os.listdir(self.root)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, np.ndarray, str]:
        image_path = self.image_path_list[index]
        image = Image.open(os.path.join(self.root, image_path))

        # process input image
        input_img = self.transform("input", image)

        # process original image
        original_img = np.array(self.transform("original", image))

        return input_img, original_img, image_path


def get_image_dataloader(config) -> DataLoader:
    # dataset
    image_dataset = ImageDataset(
        root=config.root, transform=Transforms(image_size=config.image_size)
    )

    # dataloader
    dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    return dataloader
