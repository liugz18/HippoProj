import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code

from model.ait_vitdet import AiTVitDetMlp
from model.pt_vitdet import *

def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params

class MockConfig:
    def __init__(self) -> None:
        self.dropout_prob = 0.1
        self.hidden_act = "gelu"

batch_size=1024
hidden=512
mock_config = MockConfig()
# create AIT model
ait_model = AiTVitDetMlp(in_features=hidden, hidden_features=hidden, config=mock_config)
# create AIT input Tensor
X = Tensor(
      shape=[batch_size, hidden],
      name="X",
      dtype="float16",
      is_input=True,
)
# run AIT module to generate output tensor
Y = ait_model(X)
# mark the output tensor
Y._attrs["is_output"] = True
Y._attrs["name"] = "Y"

batch_size=1024
hidden=512
# create pt model
pt_model = VitDetMlp(mock_config, hidden, hidden).cuda().half()

# create pt input
x = torch.randn([batch_size, hidden]).cuda().half()

# run pt model
pt_model.eval()
y_pt = pt_model(x)

# map pt weights to ait
weights = map_pt_params(ait_model, pt_model)

# codegen
target = detect_target()
with compile_model(
    Y, target, "./tmp", "simple_model_demo", constants=weights
) as module:
    # create storage for output tensor
    y = torch.empty([batch_size, hidden]).cuda().half()

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
