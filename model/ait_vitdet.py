import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union

from aitemplate.compiler import compile_model
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

