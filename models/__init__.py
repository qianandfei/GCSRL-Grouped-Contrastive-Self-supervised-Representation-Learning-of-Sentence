from .plus_proj_layer import Plus_Proj_layer
import torch
from .backbones import *

def get_backbone(backbone):
    if backbone=='textcnn':
        backbone=textcnn()
    else:
        NotImplementedError
    return backbone


def get_model(name, backbone):

    if name == 'GCSRL':
        model = Plus_Proj_layer(get_backbone(backbone))
    else:
        raise NotImplementedError
    return model






