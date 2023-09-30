import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code

class AiTVitDetMlp(nn.Module):
    def __init__(self, config, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, specialization="fast_gelu")
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.dropout_prob if config else 0.1) 

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

class AiTVitDetLayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over the
    channel dimension for inputs that have shape (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6, dtype="float16"):
        super().__init__()
        self.weight = nn.Parameter(shape=[normalized_shape], dtype=dtype)
        self.bias = nn.Parameter(shape=[normalized_shape], dtype=dtype)
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.dtype=dtype

    def forward(self, x):
        u = ops.reduce_mean(dim=3, keepdim=True)(x)
        s = ops.vector_norm(dim=3, keepdim=True)(x - u)
        s = s / math.sqrt(self.normalized_shape[0]) + self.eps
        x = (x - u) / s
        w = ops.unsqueeze(0)(ops.unsqueeze(0)(self.weight._tensor))
        # print(w.shape())
        x = w * x 
        x = x + ops.unsqueeze(0)(ops.unsqueeze(0)(self.bias._tensor))
        return x

class GeluActivation(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over the
    channel dimension for inputs that have shape (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, **kwargs):
        super().__init__()
        op_name = "gemm_rcr_fast_gelu"
        op_func = getattr(ops, op_name)
        self._op_name = op_name
        self.op = op_func(**kwargs)

    def forward(self, x):
        mock_weights = Tensor([x.shape()[-1], x.shape()[-1]], value=1)
        x = self.op(x, mock_weights)
        return x

class AiTVitDetResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer. It contains 3 conv layers with kernels
    1x1, 3x3, 1x1.
    """

    def __init__(self, config, in_channels, out_channels, bottleneck_channels):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            in_channels (`int`):
                Number of input channels.
            out_channels (`int`):
                Number of output channels.
            bottleneck_channels (`int`):
                Number of output channels for the 3x3 "bottleneck" conv layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, 1)
        self.norm1 = AiTVitDetLayerNorm(bottleneck_channels)
        self.act1 = GeluActivation()

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, padding=1)
        self.norm2 = AiTVitDetLayerNorm(bottleneck_channels)
        self.act2 = GeluActivation()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, 1)
        self.norm3 = AiTVitDetLayerNorm(out_channels)

    def forward(self, x):
        out = x
        # print(out.shape)
        out = ops.permute021()(ops.permute0213()(out)) # permute (0, 1, 2, 3) -> (0, 3, 1, 2)
        # print(out.shape)
        for layer in self.children():
            # print(layer)
            # print(out.shape)
            out = layer(out)

        out = ops.permute0213()(ops.permute021()(out)) # permute (0, 3, 1, 2) -> (0, 1, 2, 3)
        out = x + out
        # print(out.shape)
        return out
