# Blake Mini-Project on AIT

## Benchmarking VitMatte modules

| Module                       | Maximum Absolute Error | AIT Runtime (s)  | Pytorch Runtime (s)  |
|------------------------------|------------------------|-------------|-----------------|
| VitMatteBasicConv3x3         |   0.0005               |   0.0288    |    0.1259       |
| VitMatteConvStream           |   0.0002               |   0.3283    |    0.4357       |
| VitMatteHead                 |   0.0010               |   0.2806    |    0.4668       |
| VitMatteFusionBlock          |   0.0000               |   0.2864    |    0.3920       |
| VitMatteDetailCaptureModule  |   0.0002               |   3.3303    |    4.9254       |

I have successfully ported all the modules of VitMatte detail capture model, verified their correctness w.r.t. PyTorch GT output, and benchmarked performance difference.

Key takeaways and challenges: 

1. Setting up the entire env is time-consuming, given it's in a docker environment, which needs CUDA, nvcc, PyToch and AIT installation from source, solving a lot of dependencies.

2. Finding the corresponding ops from AIT to replace PyTorch ops, which needs siginificant effort to read and search the source code of AIT (since the docs are not complete). At first I thought there were no BatchNorm nor GELU activation, but found them later in the source code. 

3. AIT itself has a certain learning curve, but I do get a lot of momentum in the process. A lot of minor caveats like CNN input channel can't be odd; When sending input directly to output, need Identity() to avoid marking input as output in the comp graph

4. It's proven that bottom-up test driven port module by module is a right way. When I port the ConvStram at first, the error was high, because AIT nn.Conv2d is unbiased but in PyTorch default is biased. Such error can be hard to found if not the unit tests. 

5. I saved many intermediate Torch Tensors from the original huggingface pipeline, in order to do the module tests.

The VitMatte module is easier to port compared to its VitDet backbone. 

## Benchmarking VitDet modules
| Module                       | Maximum Absolute Error | AIT Runtime (s) | Pytorch Runtime (s) |
|------------------------------|------------------------|-------------|-----------------|
| VitDetMlp                    |    0.0000              | Not Recorded|  Not Recorded   |
| VitDetLayerNorm              |    0.0000              | Not Recorded|   Not Recorded  |
| VitDetResBottleneckBlock     |    0.0039              |   0.0877    |   0.4749        |
| VitDetAttention              |    0.0002              |   0.1238    |   0.2775        |
| VitDetLayer                  |    0.0156              |   0.4193    |   0.5886        |
| VitDetEncoder                |    0.0469              |   3.9350    |   10.5954       |



For the backbone part, I have finally ported VitDetEncoder (w/o positional encoding), which is the key part of the backbone. The only parts unfinished are: 

1. VitDetEmbedding: AIT doesn't support 2d interpolate with different H, W scale.

2. The relative positional encoding of VitDetEncoder: Same issue with 2d interpolate, also AIT doesn't seem to have einsum op.

Given enough time I should be able to extend the ops in AIT and find a way out, but I regret to say that I have coursework that needs much attention this week, and wasn't able to port it end2end. 

Key takeaways are:

1. This part is more challenging since the backbone has many ops unpresent in AIT, and one has to find a way to replace those with existing ops. For example AIT doesn't have 2d padding, so need to use permute + pad_last_dim.

2. Finding the corresponding ops in AIT requires more careful and thorough reading of AIT source code, for example AIT Tensor is not slicable, so have to use APIs like dynamic_slicing; matmul in AIT is given in certain APIs like bmm_rcr()

3. Unit tests still proves to be effective. One time I encountered error in VitDetResBottleneckBlock, which propagates from AITVitDetLayerNorm, in which elementwise(FuncEnum.POW)(2, x-u) is not accurate, so used (x-u) * (x-u).

In general I'd say AIT is a great framework that accelerates heavy DL models, which is super exciting to me. Though it indeed requires certain amount of warmups to get familiar with, given the docs are not complete as well as some Github issues. The learning curve can be greatly flattened if there are more examples provided in official docs.

## Reproduce Environment Record

### CUDA Compilation Tools
- **Release**: 12.2
- **Version**: V12.2.128

### Build Information
- **CUDA Version**: 12.2.r12.2
- **Compiler Build**: 33053471_0

### Package Information
- **Python**: 3.10.0
- **PyTorch**: 2.0.1+cu118
more info in environment.yml

## Run VitDet module unit tests 

```bash
python -m tests.test_VitDetMlp

python -m tests.test_VitDetLayerNorm

python -m tests.test_VitDetResBottleneckBlock

python3 -m tests.test_VitDetAttention

python3 -m tests.test_VitDetLayer

python3 -m tests.test_VitDetEncoder

```

## Run VitMatte module unit tests 

```bash

python -m tests.test_VitMatteBasicConv3x3

python -m tests.test_VitMatteConvStream

python -m tests.test_VitMatteHead

python -m tests.test_VitMatteFusionBlock

python -m tests.test_VitMatteDetailCaptureModule
```

## Repo Structure

- `environment.yml`: Conda environment specification.
- `log.md`: Project log or notes.
- `readme.md`: Project documentation and instructions.
- `run.py`: Main executable script (Only for running PyTorch module, since the porting is not end2end).

- `model`: Directory for model implementations.
  - `ait_vitdet.py`: AIT VitDet model.
  - `ait_vitmatte.py`: AIT VitMatte model.
  - `pt_vitdet.py`: PyTorch VitDet model.
  - `pt_vitmatte.py`: PyTorch VitMatte model.

- `saved_tensors`: Directory of saved intermediate tensor data, for unit tests.
  - Contains tensors like `alphas.pt`, `attention_input.pt`, etc.

- `tests`: Unit tests directory.
  - Contains tests for components such as `VitDetEncoder`, `VitMatteDetailCaptureModule`, etc.

- `try`: Directory for experimental scripts.
