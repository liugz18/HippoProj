# Timeline and thoughts for mini-proj

## Sep 28
Set up SSH and read paper about ViTMatte and ViTDet

## Sep 29
Set up Cuda, conda, torch, AIP env, read code about huggingface ViTMatte, seems like decoder easier to start with than ViTDet backbone, plan to port decoder first

## Sep 29
BatchNorm and unbiased conv not available in AIT, seems non-trivial how to circumvent this, plan to port backbone first

plan to implement modules from easy to hard: VitDetMlp, 