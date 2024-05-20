from encoder import Encoder
from decoder import Decoder
from framework.training import Module
from torch import Tensor

class FourierDiffusion(Module):
    def __init__(self,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, ) -> Tensor:
        
    def _default_training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        out = self(I_gt * mask_in)
        return out # revisar

    def validation_step(self, batch: Tensor, idx: int) -> None:
        