import torch
from torch import nn, Tensor
from typing import Callable
from framework.nn.fourier import _FourierConv
from framework.nn.complex import (
    ComplexSiLU,
    ComplexUpsampling
)

def ResidualConnection(x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
    return x + sublayer(x)


class SingleBlock(nn.Sequential):
    def __init__(self, activation: str) -> None:
        super().__init__(

        )
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

class ResidualBlock(nn.Module):
    # c, h, w -> c, h, w 
    def __init__(self, layers: int, activation: str) -> None:
        super().__init__()
        
        self.layers = layers
        for layer in range(self.layers):
            setattr(self, f"layer{layer}", nn.Sequential(
                SingleBlock(activation)
            ))

    def forward(self, x: Tensor) -> Tensor:
        for layer in range(self.layers):
            x = ResidualConnection(x, getattr(self, f'layer{layer}'))
        return x
    
    
class Upsampling(nn.Module): 
    def __init__(self, channels: int , height: int, width: int, scale_factor: int = 2) -> None:
        self.upsample = ComplexUpsampling(scale_factor)
        self.conv = _FourierConv(channels, height, width)
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.upsample(x))

class Downsampling(nn.Module):
    def __init__(self, channels: int, height: int, width: int, scale_factor: int = 2) -> None:
        self.pool = nn.AvgPool2d(scale_factor, scale_factor)
        self.conv = _FourierConv(channels, height, width)        

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))
    
                