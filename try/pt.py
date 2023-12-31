import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code


class PTSimpleModel(torch.nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = torch.nn.Linear(hidden, 4 * hidden)
        self.act1 = torch.nn.functional.gelu
        self.dense2 = torch.nn.Linear(4 * hidden, hidden)
        self.layernorm = torch.nn.LayerNorm(hidden, eps=eps)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.act1(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = hidden_states + input
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class AITSimpleModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = nn.Linear(hidden, 4 * hidden, specialization="fast_gelu")
        self.dense2 = nn.Linear(4 * hidden, hidden)
        self.layernorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.dense2(hidden_states)
        hidden_states = hidden_states + input
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params


batch_size = 1024
hidden = 512
# create pt model
pt_model = PTSimpleModel(hidden).cuda().half()

# create pt input
x = torch.randn([batch_size, hidden]).cuda().half()

# run pt model
pt_model.eval()
y_pt = pt_model(x)

# map
