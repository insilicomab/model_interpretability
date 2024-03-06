import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class ImageDataset(Dataset):
    def __init__(self, root: str, transform: Compose) -> None:
        self.root = root
        self.image_path_list = [f for f in os.listdir(self.root)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_path_list[index]
        image = Image.open(os.path.join(self.root, image_path))

        # process input image
        input_img = self.transform("input", image)

        # process original image
        original_img = self.transform("original", image)

        return input_img, original_img, image_path
