import torch
from torch import nn
import warnings


from typing import Any, Optional, Callable
from functools import partial
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.ssdlite import _normal_init, _mobilenet_extractor, SSDLiteHead
from torchvision.models.utils import load_state_dict_from_url

from .ssd import SSD
from .shakedrop.backbone import mobilenet_v3_large

__all__ = ['ssdlite320_mobilenet_v3_large', 'ssdlite320_mobilenet_v3_large_shakedrop']

model_urls = {
    "ssdlite320_mobilenet_v3_large_coco": "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
}

def ssdlite320_mobilenet_v3_large(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                                  **kwargs: Any):
    """Constructs an SSDlite model with input size 320x320 and a MobileNetV3 Large backbone, as described at
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_ and
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    See :func:`~torchvision.models.detection.ssd300_vgg16` for more details.

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
    """
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6)

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = _mobilenet_extractor("mobilenet_v3_large", progress, pretrained_backbone, trainable_backbone_layers,
                                    norm_layer, reduced_tail=reduce_tail, **kwargs)

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, -1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, size, num_classes,
                head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer), **kwargs)

    if pretrained:
        weights_name = 'ssdlite320_mobilenet_v3_large_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return model



def ssdlite320_mobilenet_v3_large_shakedrop(
        pretrained: bool = True,
        progress: bool = True,
        num_classes: int = 91,
        pretrained_backbone: bool = False,
        trainable_backbone_layers: Optional[int] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6
    )

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet_v3_large(
        pretrained=pretrained_backbone, progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs
    )
    if not pretrained_backbone:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )

    if pretrained:
        weights_name = "ssdlite320_mobilenet_v3_large_coco"
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return