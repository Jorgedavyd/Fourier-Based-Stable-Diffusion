from framework import _FourierConvNd
from torch import nn
from torch import Tensor

class Decoder(nn.Sequential): 
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Tensor) -> Tensor:
