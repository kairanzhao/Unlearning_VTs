import timm
import torch
import torch.nn as nn

__all__ = [
    "SwinTransformer",
    "swin_tiny",
]

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Convert lists to tensors if needed
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        # Normalize assuming channel is dim 1
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return (tensor - mean) / std

class SwinTransformer(nn.Module):
    """
    A simple wrapper around a timm Swin Transformer model that always applies
    a normalization layer as the first step in the forward pass.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        # Always add the normalization layer with CIFAR-10 statistics
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First, normalize the input
        x = self.normalize(x)
        # Then, pass the normalized tensor through the model
        return self.model(x)

def swin_tiny(num_classes: int, pretrained: bool = True, **kwargs) -> SwinTransformer:
    """
    Constructs a Swin Transformer Tiny model using timm.
    
    Args:
        num_classes (int): Number of classes for the classification head.
        pretrained (bool): If True, loads pretrained weights.
        **kwargs: Additional keyword arguments passed to timm.create_model.
    
    Returns:
        SwinTransformer: A wrapper model that applies normalization followed by the timm model.
    """
    model = timm.create_model(
    'swin_tiny_patch4_window7_224',  # Base model template (parameters will be overridden)
    pretrained=pretrained,
    img_size=32, 
    num_classes=num_classes, 
    **kwargs
    )
    return SwinTransformer(model)
