import torch
from torch import nn

from train import SurfaceNetV2


class ReflectedSurfaceNet(nn.Module):
    """Average registered logits from the four rectilinear orientations."""

    def __init__(self, model: SurfaceNetV2, context_weight: float):
        super().__init__()
        self.model = model
        self.context_weight = context_weight

    def forward(self, pixels):
        views = torch.cat((pixels, pixels.flip((-1,)), pixels.flip((-2,)), pixels.flip((-1, -2))))
        masks, context = self.model.outputs(views)
        masks = masks + self.context_weight * context[:, :, None, None]
        original, horizontal, vertical, both = masks.chunk(4)
        return (original + horizontal.flip((-1,)) + vertical.flip((-2,)) + both.flip((-1, -2))) * .25
