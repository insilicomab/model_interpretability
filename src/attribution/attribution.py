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


class GradcamAttribution(BaseAttribution):
    def __init__(
        self, model: nn.Module, layer: nn.Module, device_ids: int = None
    ) -> None:
        super().__init__(model)
        self.__layer = layer
        self.__gradcam = captum.attr.LayerGradCam(self._model, self.__layer, device_ids)

    @property
    def layer(self):
        return self.__layer

    def attribute(
        self,
        inputs: torch.Tensor,
        target: int,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ):
        attribution = self.__gradcam.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            relu_attributions=relu_attributions,
        )
        attribution_img = attribution[0].cpu().permute(1, 2, 0).detach().numpy()
        return attribution_img


class GuidedGradcamAttribution(BaseAttribution):
    def __init__(
        self, model: nn.Module, layer: nn.Module, device_ids: int = None
    ) -> None:
        super().__init__(model)
        self.__layer = layer
        self.__guided_gradcam = captum.attr.GuidedGradCam(
            self._model, self.__layer, device_ids
        )

    @property
    def layer(self):
        return self.__layer

    def attribute(
        self,
        inputs: torch.Tensor,
        target: int,
        additional_forward_args: Any = None,
        interpolate_mode: str = "nearest",
        attribute_to_layer_input: bool = False,
    ):
        attribution = self.__guided_gradcam.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            interpolate_mode=interpolate_mode,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        attribution_img = attribution[0].cpu().permute(1, 2, 0).detach().numpy()
        return attribution_img


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
