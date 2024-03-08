import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


class TimmNet(nn.Module):
    def __init__(self, config: DictConfig):
        super(TimmNet, self).__init__()
        self.config = config
        self.model_name = config.model_name
        self.num_classes = config.num_classes
        self.pretrained = False
        self.net = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
        )

    def forward(self, x):
        return self.net(x)


def get_model(config: DictConfig) -> torch.nn.Module:
    _model_state_dict = torch.load(config.model_path, map_location=torch.device("cpu"))[
        "state_dict"
    ]
    model_state_dict = {
        k.replace("model.", ""): v for k, v in _model_state_dict.items()
    }
    model = TimmNet(config)
    model.load_state_dict(model_state_dict, strict=True)
    return model.net.eval()
