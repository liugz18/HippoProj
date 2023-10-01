# Timeline and thoughts for mini-proj

## Sep 28
Set up SSH and read paper about ViTMatte, seems like the model comprises 2 parts: A backbone ViTDet model and a decoder made of Conv stream 

## Sep 29
Set up Cuda, conda, torch, AIT env, read code about huggingface ViTMatte, seems like decoder easier to start with than ViTDet backbone, plan to port decoder first

## Sep 29
For decoder, batchNorm and unbiased conv are not available in AIT, AIT official demo of ResNet-50 is not easy to read, seems non-trivial how to circumvent this, plan to port backbone first instead

Plan to implement modules in a bottom up approach, from easy to hard: VitDetMlp, VitDetLayerNorm, VitDetDropPath, VitDetAttention, VitDetEmbeddings, etc

Saved intermediate tensors on disk for modularized porting

Moved pytorch vitdet and vitmatte model to local dir for easy hacking

## Sep 30



compiling a conv2d fails out of unknown reason ("tmp_key = next(iter(self._attrs["op_instance"].keys()))
StopIteration" during profiling)

Ported AiTVitDetResBottleneckBlock, seems like AIT Conv2d can't compile when in_channels is odd

Found BatchNorm, GELU Activation in the [source code of AIT](https://github.com/facebookincubator/AITemplate/blob/d5d0acd4fd1aed1c316a5860a2bf6425483df4e1/python/aitemplate/frontend/nn/activation.py), so plan to work on porting decoder, which may be easier to port

Ported VitMatteBasicConv3x3 and AITVitMatteConvStream, unit test met unexpected input bug

Solved unexpected input bug, need to use nn.identity() to avoid marking input as output, mid layer still have precision issue