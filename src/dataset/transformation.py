from torchvision import transforms


class Transforms:
    """
    A class that defines image transformations for different phases (e.g., input, original).

    Args:
        image_size (int): The size to which images are resized.

    Attributes:
        image_size (int): The size to which images are resized.
        transform (dict): A dictionary containing transformation functions for different phases.

    Methods:
        __call__(self, phase, img):
            Apply the specified transformation to the input image based on the given phase.

    Examples:
        transforms = Transforms(image_size=224)
        input_img = transforms("input", original_img)
    """

    def __init__(self, image_size: int):
        """
        Initializes the Transforms object with the provided image size.

        Args:
            image_size (int): The size to which images are resized.
        """
        self.image_size = image_size
        self.transform = {
            "input": transforms.Compose(
                [
                    transforms.CenterCrop([self.image_size, self.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "original": transforms.Compose(
                [
                    transforms.CenterCrop([self.image_size, self.image_size]),
                    transforms.ToTensor(),
                ]
            ),
        }

    def __call__(self, phase, img):
        """
        Apply the specified transformation to the input image based on the given phase.

        Args:
            phase (str): The phase for which the transformation should be applied (e.g., "input", "original").
            img (PIL.Image): The input image to be transformed.

        Returns:
            torch.Tensor: The transformed image as a PyTorch tensor.
        """
        return self.transform[phase](img)
