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

    def __init__(self, normalized_shape, eps=1e-6, dtype="float16",):
        super().__init__()
        self.weight = nn.Parameter(shape=[normalized_shape], dtype=dtype)
        self.bias = nn.Parameter(shape=[normalized_shape], dtype=dtype)
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.dtype=dtype

    def forward(self, x):
        u = ops.reduce_mean(dim=1, keepdim=True)(x)
        s = ops.vector_norm(dim=1, keepdim=True)(x - u)
        s = s / math.sqrt(self.normalized_shape[0]) + self.eps
        x = (x - u) / s
        x = ops.unsqueeze(2)(ops.unsqueeze(1)(self.weight._tensor)) * x + ops.unsqueeze(2)(ops.unsqueeze(1)(self.bias._tensor))
        return x