from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
# from .vision_transformer import *
from .swin_transformer import *
from .vit_transformer import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn, # vgg16 with batch normalization
    "vgg16_bn_lth": vgg16_bn_lth, # vgg16 with bn and Lottery Ticket Hypothesis
    # "vit_b_16": vit_b_16,
    # "vit_b_32": vit_b_32,
    # "vit_custom": vit_custom,
    "swin_tiny": swin_tiny,
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
}
