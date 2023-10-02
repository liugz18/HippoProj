import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code

from model.ait_vitdet import AITVitDetLayerNorm
from model.pt_vitdet import *

def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        
        assert name in pt_params
        param = pt_params[name]
        print(ait_name, param)
        mapped_pt_params[ait_name] = param
    return mapped_pt_params

class MockConfig:
    def __init__(self) -> None:
        self.dropout_prob = 0.1
        self.hidden_act = "gelu"

batch_size=1
hidden=5
shape_pt = [batch_size, 3, hidden, hidden]
shape = [batch_size, hidden, hidden, 3]
mock_config = MockConfig()
# create AIT model
ait_model = AITVitDetLayerNorm(3)
# create AIT input Tensor
X = Tensor(
      shape=shape,
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
pt_model = VitDetLayerNorm(3).cuda().half()

# create pt input
x = torch.randn(shape_pt).cuda().half()

# run pt model
pt_model.eval()
y_pt = pt_model(x)
y_pt = y_pt.permute(0, 2, 3, 1).contiguous()

# map pt weights to ait
weights = map_pt_params(ait_model, pt_model)

# codegen
target = detect_target()
print(target)
with compile_model(
    Y, target, "./tmp", "VitDetLayerNorm", constants=weights
) as module:
    # create storage for output tensor
    y = torch.empty(shape).cuda().half()

    # inputs and outputs dict
    x = x.permute(0, 2, 3, 1).contiguous()
    inputs = {"X": x}
    outputs = {"Y": y}

    # run
    module.run_with_tensors(inputs, outputs, graph_mode=True)

    # verify output is correct
    print(y - y_pt, (y-y_pt).max())
    print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

