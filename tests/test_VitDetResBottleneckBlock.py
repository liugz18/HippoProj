import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from collections import OrderedDict

from model.ait_vitdet import AITVitDetResBottleneckBlock
from model.pt_vitdet import *


def map_pt_params(ait_model: nn.Module, pt_model):
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")

        assert name in pt_params, f"{name} {pt_params.keys()}"
        params = pt_params[name]
        print(name, params.shape)
        if len(params.shape) == 4:
            # NCHW->NHWC
            params = params.permute((0, 2, 3, 1)).contiguous()

        mapped_pt_params[ait_name] = params
    return mapped_pt_params


class MockConfig:
    def __init__(self) -> None:
        self.dropout_prob = 0.1
        self.hidden_act = "gelu"


batch_size = 1
hidden = 128
shape_pt = [batch_size, 4, hidden, hidden]
shape = [batch_size, hidden, hidden, 4]
mock_config = MockConfig()
# create AIT model
ait_model = AITVitDetResBottleneckBlock(None, 4, 4, 6)
ait_model.name_parameter_tensor()
# create AIT input Tensor
X = Tensor(
    shape=shape_pt,
    name="X",
    dtype="float16",
    is_input=True,
)
# run AIT module to generate output tensor
Y = ait_model(X)
# mark the output tensor
Y._attrs["is_output"] = True
Y._attrs["name"] = "Y"


# create pt model
pt_model = VitDetResBottleneckBlock(mock_config, 4, 4, 6).cuda().half()

# create pt input
x = torch.randn(shape_pt).cuda().half()

# run pt model
pt_model.eval()
y_pt = pt_model(x)

# map pt weights to ait
weights = map_pt_params(ait_model, pt_model)

# codegen
target = detect_target()


module = compile_model(
    Y, target, "./tmp", "AITVitDetResBottleneckBlock", constants=weights
)

y = torch.empty(shape_pt).cuda().half()

# inputs and outputs dict
inputs = {"X": x}
outputs = {"Y": y}

# run
module.run_with_tensors(inputs, outputs, graph_mode=True)

# verify output is correct
print(y - y_pt, (y - y_pt).max())
print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

# benchmark ait and pt
count = 1000
ait_t, _, _ = module.benchmark_with_tensors(
    inputs, outputs, graph_mode=True, count=count
)
print(f"AITemplate time: {ait_t} ms/iter")

pt_t = benchmark_torch_function(count, pt_model.forward, x)
print(f"PyTorch eager time: {pt_t} ms/iter")
