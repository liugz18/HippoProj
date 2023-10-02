import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from collections import OrderedDict

from model.ait_vitmatte import AITVitMatteBasicConv3x3
from model.pt_vitmatte import *


def map_pt_params(ait_model:nn.Module, pt_model):
    pt_params = dict(pt_model.named_parameters())
    # from IPython import embed; embed()
    pt_params['batch_norm.running_mean'] = pt_model.batch_norm.running_mean.clone()
    pt_params['batch_norm.running_var'] = pt_model.batch_norm.running_var.clone()
    pt_params['batch_norm.num_batches_tracked'] = pt_model.batch_norm.num_batches_tracked.clone()
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
        self.batch_norm_eps = 1e-6

batch_size=1
hidden=128
in_channels=32
out_channels=16
shape_pt = [batch_size, in_channels, hidden, hidden]
shape_after = [batch_size, out_channels, hidden//2, hidden//2]
shape = [batch_size, hidden, hidden, 4]
mock_config = MockConfig()
# create AIT model
ait_model = AITVitMatteBasicConv3x3(mock_config, in_channels=in_channels, out_channels=out_channels, stride=2, padding=1)
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
pt_model = VitMatteBasicConv3x3(mock_config, in_channels=in_channels, out_channels=out_channels, stride=2, padding=1).cuda().half()

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

y = torch.empty(shape_after).cuda().half()

# inputs and outputs dict
inputs = {"X": x}
outputs = {"Y": y}

# run
module.run_with_tensors(inputs, outputs, graph_mode=True)

# verify output is correct
print(y - y_pt, (y-y_pt).max())
print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

# benchmark ait and pt
count = 1000
ait_t, _, _ = module.benchmark_with_tensors(
    inputs, outputs, graph_mode=True, count=count
)
print(f"AITemplate time: {ait_t} ms/iter")

pt_t = benchmark_torch_function(count, pt_model.forward, x)
print(f"PyTorch eager time: {pt_t} ms/iter")
