from abc import ABC, abstractmethod
from typing import Any

import captum
import torch
import torch.nn as nn


class BaseAttribution(ABC):
    def __init__(self, model: nn.Module) -> None:
        self._model = model

    @property
    def model(self):
        return self._model

    @abstractmethod
    def attribute(self):
        pass


class SaliencyAttribution(BaseAttribution):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.__saliency = captum.attr.Saliency(self._model)

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
        return attribution_img
