import aitemplate.testing.detect_target as detect_target
from collections import OrderedDict
from aitemplate.frontend import nn, Tensor
from aitemplate.compiler import compile_model
import torch

class TorchModule(torch.nn.Module):
    def __init__(self):
        super(TorchModule, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(4,8,4,2)
        self.deconv1 = torch.nn.ConvTranspose2d(8,8,4,2)


    def forward(self, x):
        x = self.deconv(x)
        x = self.deconv1(x)

        return x

class AitModule(nn.Module):
    def __init__(self):
        super(AitModule, self).__init__()
        self.deconv = nn.ConvTranspose2dBias(4,8,4,2)
        self.deconv1 = nn.ConvTranspose2dBias(8,8,4,2)

    def forward(self, x):
        x = self.deconv(x)
        x = self.deconv1(x)

        return x

def map_pt_params(ait_model:nn.Module, pt_model):
  pt_params = dict(pt_model.named_parameters())
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

pt_model = TorchModule()
torch.nn.init.xavier_normal_(pt_model.deconv.weight)
torch.nn.init.xavier_normal_(pt_model.deconv1.weight)
pt_model.cuda().half()
pt_model.eval()



x = torch.randn([1, 4, 128, 128]).cuda().half()
with torch.no_grad():
    y_pt = pt_model(x)

ait_model = AitModule()
weights = map_pt_params(ait_model, pt_model)


X = Tensor(
      shape=[1, 128, 128, 4],
      name="x",
      dtype="float16",
      is_input=True,
)

# run AIT module to generate output tensor
Y = ait_model(X)
# mark the output tensor
Y._attrs["is_output"] = True
Y._attrs["name"] = "y"

target = detect_target()
with compile_model(Y, target, "./tmp", "test", constants=weights) as module:
    y = torch.empty([1, y_pt.shape[-1], y_pt.shape[-2], y_pt.shape[1]]).cuda().half()

    x = x.permute((0, 2, 3, 1)).contiguous()

    # inputs and outputs dict
    inputs = {"x": x }
    outputs = {"y": y }

    # run
    module.run_with_tensors(inputs, outputs, graph_mode=True)

    y_ait = y.permute((0, 3, 1, 2))

    print(y_ait.shape, torch.allclose(y_ait, y_pt, atol=1e-2, rtol=1e-2), y_ait.min(), y_pt.min(), y_ait.max(), y_pt.max())