import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from collections import OrderedDict

from model.ait_vitmatte import AITVitMatteHead, AITVitMatteConfig
from model.pt_vitmatte import *


def mark_output(y):
    outputs = ()
    y._attrs["is_output"] = True
    y._attrs["name"] = f"output"
    y_shape = [d._attrs["values"][0] for d in y._attrs["shape"]]
    print("output shape: {}".format(y_shape))
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
            # print(name, param_value.shape)
            pt_params[name] = param_value.clone()
    # print(pt_params.keys())
    # print(list(ait_model.named_parameters()))

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


batch_size = 1
hidden = 640
shape_pt = [batch_size, 32, hidden, hidden]
mock_config = AITVitMatteConfig
# create AIT model
ait_model = AITVitMatteHead(mock_config)
ait_model.name_parameter_tensor()

# create pt model
pt_model = VitMatteHead(mock_config).cuda().half()

# create pt input
x = torch.randn(shape_pt).cuda().half()

# run pt model
pt_model.eval()
y_pt = pt_model(x)

# map pt weights to ait
weights = map_pt_params(ait_model, pt_model)

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

outputs = mark_output(Y)


# codegen
target = detect_target()


module = compile_model(Y, target, "./tmp", "AITVitMatteHead", constants=weights)


# inputs and outputs dict
inputs = {"X": x}

# run
module.run_with_tensors(inputs, outputs, graph_mode=True)

# verify output is correct
y = outputs[0]
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
