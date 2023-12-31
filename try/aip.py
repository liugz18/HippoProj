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
# create AIT model
ait_model = AITSimpleModel(hidden)
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

batch_size = 1024
hidden = 512
# create pt model
pt_model = PTSimpleModel(hidden).cuda().half()

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


# export CUDA_HOME=/usr/local/cuda-12.2
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export PATH=$CUDA_HOME/bin:$PATH
