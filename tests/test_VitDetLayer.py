import torch
import os
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from collections import OrderedDict

from model.ait_vitdet import AITVitDetLayer, AITVitDetConfig
from model.pt_vitdet import *


def mark_output(y):
    outputs = ()
    if type(y) != tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = f"output_{i}"
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))
        outputs += (torch.empty(y_shape).cuda().half(),)
    return outputs


def map_pt_params(ait_model, pt_model):
    pt_params = dict(pt_model.named_parameters())

    # Add the missing batch_norm statistics to pt_params
    for name, _ in ait_model.named_parameters():
        # print(name, name not in pt_params, "batch_norm" in name)
        if name not in pt_params and (
            "running_mean" in name
            or "running_var" in name
            or "num_batches_tracked" in name
        ):
            attr_sequence = name.split(".")
            param_value = pt_model
            for attr in attr_sequence:
                param_value = getattr(param_value, attr)
            print(name, param_value.shape)
            pt_params[name] = param_value.clone()
    # print(pt_params.keys())
    # print(list(ait_model.named_parameters()))

    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")

        assert name in pt_params, f"{name} {pt_params.keys()}"
        params = pt_params[name]

        if len(params.shape) == 4:
            # NCHW->NHWC
            params = params.permute((0, 2, 3, 1)).contiguous()

        mapped_pt_params[ait_name] = params

    return mapped_pt_params


def compile(ait_model, shape_1, weights):
    ait_model.name_parameter_tensor()
    # create AIT input Tensor
    X1 = Tensor(
        shape=shape_1,
        name="X1",
        dtype="float16",
        is_input=True,
    )
    # run AIT module to generate output tensor
    Y = ait_model(X1)
    # mark the output tensor
    outputs = mark_output(Y)
    module = compile_model(Y, target, "./tmp", "VitDetLayer", constants=weights)
    # y = torch.empty(shape_after).cuda().half()

    return module, outputs


mock_config = AITVitDetConfig
mock_config.use_relative_position_embeddings = False
# create pt input
batch_size = 1
use_residual_block = False
shape_1 = [batch_size, 384, 40, 60]


# codegen
target = detect_target()
# create AIT model
ait_model = AITVitDetLayer(mock_config, use_residual_block=use_residual_block)

# create pt model
pt_model = VitDetLayer(mock_config, use_residual_block=use_residual_block).cuda().half()

# map pt weights to ait
weights = map_pt_params(ait_model, pt_model)

module, outputs = compile(ait_model, shape_1, weights)


# Relative path to the .pt file
tensor_path = os.path.join(
    os.path.dirname(__file__), "..", "saved_tensors", "features.pt"
)

# Load the tensor
x1 = torch.load(tensor_path)

# Transfer to GPU and convert to half precision
x1 = x1.cuda().half()


# run pt model
pt_model.eval()
y_pt = pt_model(x1)


# inputs and outputs dict
inputs = {"X1": x1}
# run
module.run_with_tensors(inputs, outputs, graph_mode=True)

# verify output is correct
print(len(outputs), len(y_pt))
for y, y_pt in zip(outputs, y_pt):
    print("Maximum Absolute Error: ", (y - y_pt).max())
    print("Error is below threshold: ", torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))


# benchmark ait and pt
count = 1000
ait_t, _, _ = module.benchmark_with_tensors(
    inputs, outputs, graph_mode=True, count=count
)
print(f"AITemplate time: {ait_t} ms/iter")

pt_t = benchmark_torch_function(count, pt_model.forward, x1)
print(f"PyTorch eager time: {pt_t} ms/iter")
