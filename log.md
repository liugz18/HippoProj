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

