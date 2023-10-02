import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import nn as ann
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target



class AITConv2d(ann.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ann.Conv2d(in_channels, out_channels, 1, 1, 1)

    def forward(self, x):
        out = x
        out = ops.permute021()(ops.permute0213()(out)) # permute (0, 1, 2, 3) -> (0, 3, 1, 2)
        out = self.conv1(out)
        out = ops.permute0213()(ops.permute021()(out)) # permute (0, 3, 1, 2) -> (0, 1, 2, 3)
        return out

class PTConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = x
        out = self.conv1(out)

        return out

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

def compile_conv2d_module(model_name="AITConv2d", batch_size=1, **kwargs):


    model_name = f"{model_name}_{batch_size}"
    target = detect_target(**kwargs)
    # Create input tensor, need to specify the shape, dtype and is_input flag
    x = Tensor(
        shape=[batch_size, 6, 10, 10], dtype="float16", name="input0", is_input=True
    )
    model = AITConv2d(6, 8)
    # Mark all parameters with name same to PyTorch name convention
    model.name_parameter_tensor()
    # Forward the input tensor to the model, get output tensor
    y = model(x)
    # Mark output tensor
    mark_output(y)
    # Compile the model
    module = compile_model(y, target, "./tmp", model_name)
    return module



module = compile_conv2d_module()
