"""
Model factory for image classification.

Supported models
----------------
- vgg16
- resnet50
- efficientnet_b0
- convnext_tiny
"""
import logging

from torchvision import models 
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model(cfg) -> nn.Module:
    """
    Build and return a classification model configured from a dictionary.

    Loads a torchvision model with optional pretrained ImageNet weights and
    replaces the final classification layer to match the target number of
    classes.

    Parameters
    ----------
    cfg : dict
        Model configuration. Expected keys:

        - ``name`` : str
            Model name. Supported values are ``vgg16``, ``resnet50``,
            ``efficientnet_b0`` and ``convnext_tiny``.
        - ``num_classes`` : int
            Number of output classes for the final classification layer.
        - ``pretrained`` : bool, default=False
            If ``True``, loads ImageNet pretrained weights for all layers
            except the final classification head.

    Returns
    -------
    nn.Module
        Model with the final layer replaced to output ``num_classes`` logits.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported model names.

    Notes
    -----
    Each architecture exposes its classification head differently:

    - ``vgg16``: ``model.classifier[6]`` — last layer of a 7-layer Sequential.
    - ``resnet50``: ``model.fc`` — single Linear layer.
    - ``efficientnet_b0``: ``model.classifier[1]`` — last layer of a 2-layer Sequential.
    - ``convnext_tiny``: ``model.classifier[2]`` — last layer of a 3-layer Sequential.

    Examples
    --------
    >>> model = get_model({"name": "resnet50", "num_classes": 25, "pretrained": True})
    >>> model = get_model({"name": "vgg16", "num_classes": 10})
    """
    name = cfg["name"]
    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", False)

    match name:
        case "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            
        case "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        case "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        case "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
            
        case _:
            raise ValueError(f"Unknown model: '{name}'")
        
    logger.info(f"Model {name} with {num_classes} classes created succesfully.")
    
    return model
