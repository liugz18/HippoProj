import torch
from transformers.activations import ACT2FN
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code

from model.ait_vitdet import GeluActivation
from model.pt_vitdet import *

def mark_output(y):
    """Different to PyTorch, we need to explicit mark output tensor for optimization,

    Parameters
    ----------
    y : List[Tensor]
        List of output tensors
    """
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))

def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        print(ait_name)
        assert name in pt_params
        mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params

class MockConfig:
    def __init__(self) -> None:
        self.dropout_prob = 0.1
        self.hidden_act = "gelu"

class PtGeluActivation(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.act1 = ACT2FN[config.hidden_act]


    def forward(self, x):
        out = x
        out = self.act1(out)
        return out

batch_size=1
hidden=10
# shape = [batch_size, 3, hidden, hidden]
shape = [batch_size, hidden, hidden, 3]
shape_pt = [batch_size, 3, hidden, hidden]
mock_config = MockConfig()
# create AIT model
ait_model = GeluActivation()
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
# mark_output(Y)



# create pt model
pt_model = PtGeluActivation(mock_config).cuda().half()

# create pt input

x = torch.randn(shape_pt).cuda().half()

# run pt model
pt_model.eval()
y_pt = pt_model(x)
# y_pt = torch.permute(y_pt, [0,2,3,1])
# y_pt = y_pt.contiguous()

# map pt weights to ait
weights = map_pt_params(ait_model, pt_model)

# codegen
target = detect_target()
print("Target:", target)
# with compile_model(
#     Y, target, "./tmp", "simple_model_demo_3", constants=weights
# ) as module:

module = compile_model(
    Y, target, "./tmp", "GeluActivation"
) 

# for name, param in ait_params.items():
#     module.set_constant_with_tensor(name, param)
    # create storage for output tensor
y = torch.empty(shape_pt).cuda().half()

# inputs and outputs dict
inputs = {"X": x}
outputs = {"Y": y}

# run
module.run_with_tensors(inputs, outputs, graph_mode=True)

# verify output is correct
print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

# benchmark ait and pt
count = 1000
ait_t, _, _ = module.benchmark_with_tensors(
    inputs, outputs, graph_mode=True, count=count
)
print(f"AITemplate time: {ait_t} ms/iter")

pt_t = benchmark_torch_function(count, pt_model.forward, x)
print(f"PyTorch eager time: {pt_t} ms/iter")
