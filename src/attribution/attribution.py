from abc import ABC, abstractmethod
from typing import Any

import captum
import torch
import torch.nn as nn


class BaseAttribution(ABC):
    def __init__(self, model: nn.Module, int_to_label: dict) -> None:
        self._model = model
        self._int_to_label = int_to_label

    @property
    def model(self):
        return self._model

    @property
    def int_to_label(self):
        return self._int_to_label

    @abstractmethod
    def attribute(self):
        pass


class SaliencyAttribution(BaseAttribution):
    def __init__(self, model: nn.Module, int_to_label: dict) -> None:
        super().__init__(model, int_to_label)
        self.__saliency = captum.attr.Saliency(self._model)
        self.__attribution_name = "Saliency"

    def attribute(
        self,
        inputs: torch.Tensor,
        target: int,
        abs: bool = True,
        additional_forward_args: Any = None,
    ):
        attribution = self.__saliency.attribute(
            inputs=inputs,
            target=target,
            abs=abs,
            additional_forward_args=additional_forward_args,
        )
        attribution_img = attribution[0].cpu().permute(1, 2, 0).detach().numpy()
        target_class = self._int_to_label[target]
        return attribution_img, target_class, self.__attribution_name
