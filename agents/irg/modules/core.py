import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np

from agents.irg.modules.vae_backbone import *

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight) #he normal
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BaseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone_mapping = {
            4 : {
                "encoder" : SimpleEncoder,
                "decoder" : SimpleDecoder
            },
            2 : {
                "encoder" : SimpleEncoder2PlayerSmall,
                "decoder" : SimpleDecoder2PlayerSmall
            }
        }
    
    def forward(self):
        raise NotImplementedError
    
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.kaiming_uniform_(layer.weight) #he normal
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer