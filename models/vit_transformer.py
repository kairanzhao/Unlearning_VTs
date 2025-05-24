import timm
import torch
import torch.nn as nn

__all__ = [
    "VitTransformer",
    "vit_tiny",
    "vit_small",
]

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Convert lists to tensors if needed.
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        # Normalize assuming channel is dim=1.
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return (tensor - mean) / std

class VitTransformer(nn.Module):
    """
    A simple wrapper around a timm Vision Transformer (ViT) model that always applies
    a normalization layer as the first step in the forward pass.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        # Always add the normalization layer using CIFAR-10 statistics.
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply normalization first.
        x = self.normalize(x)
        # Then pass through the base Vision Transformer model.
        return self.model(x)

def vit_tiny(num_classes: int, pretrained: bool = True, **kwargs) -> VitTransformer:
    """
    Constructs a Vision Transformer (ViT) Tiny model using timm.
    
    Args:
        num_classes (int): Number of classes for the classification head.
        pretrained (bool): If True, loads pretrained weights.
        **kwargs: Additional keyword arguments passed to timm.create_model.
    
    Returns:
        VitTransformer: A wrapper model that applies normalization followed by the timm ViT model.
    """
    model = timm.create_model(
        "vit_tiny_patch8_224",  # timm model name for ViT Tiny.
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=32,  # Set to 32 if you're adapting for CIFAR-10; otherwise use 224.
        **kwargs
    )
    return VitTransformer(model)

def vit_small(num_classes: int, pretrained: bool = True, **kwargs) -> VitTransformer:
    """
    Constructs a Vision Transformer (ViT) Tiny model using timm.
    
    Args:
        num_classes (int): Number of classes for the classification head.
        pretrained (bool): If True, loads pretrained weights.
        **kwargs: Additional keyword arguments passed to timm.create_model.
    
    Returns:
        VitTransformer: A wrapper model that applies normalization followed by the timm ViT model.
    """
    model = timm.create_model(
        "vit_small_patch8_224",  
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=32,  # Set to 32 if you're adapting for CIFAR-10; otherwise use 224.
        **kwargs
    )
    return VitTransformer(model)
