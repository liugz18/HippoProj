# coding=utf-8
# Copyright 2023 HUST-VL and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch ViTMatte model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn as tnn

# Absolute imports from the `transformers` library
from transformers import AutoBackbone
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils.backbone_utils import BackboneMixin
from transformers.models.vitmatte.configuration_vitmatte import VitMatteConfig

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from aitemplate.compiler.public import elementwise, FuncEnum


VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hustvl/vitmatte-small-composition-1k",
    # See all VitMatte models at https://huggingface.co/models?filter=vitmatte
]


# General docstring
# _CONFIG_FOR_DOC = "AITVitMatteConfig"
configs = {
    "_commit_hash": "03f5646d1ed954c462f3837123fba723dfd1b3d5",
    "_name_or_path": "hustvl/vitmatte-small-composition-1k",
    "architectures": ["VitMatteForImageMatting"],
    "backbone_config": {
        "hidden_size": 384,
        "image_size": 512,
        "model_type": "vitdet",
        "num_attention_heads": 6,
        "num_channels": 4,
        "out_features": ["stage12"],
        "out_indices": [12],
        "residual_block_indices": [2, 5, 8, 11],
        "use_relative_position_embeddings": True,
        "window_block_indices": [0, 1, 3, 4, 6, 7, 9, 10],
        "window_size": 14,
    },
    "batch_norm_eps": 1e-05,
    "convstream_hidden_sizes": [48, 96, 192],
    "fusion_hidden_sizes": [256, 128, 64, 32],
    "hidden_size": 384,
    "initializer_range": 0.02,
    "model_type": "vitmatte",
    "torch_dtype": "float32",
    #   "transformers_version": null
}
AITVitMatteConfig = VitMatteConfig(**configs)


@dataclass
class ImageMattingOutput(ModelOutput):
    """
    Class for outputs of image matting models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss.
        alphas (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           Estimated alpha values.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    alphas: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# class AITVitMattePreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = AITVitMatteConfig
#     main_input_name = "pixel_values"
#     supports_gradient_checkpointing = True

#     def _init_weights(self, module):
#         if isinstance(module, tnn.Conv2d):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()

#     def _set_gradient_checkpointing(self, module, value=False):
#         if isinstance(module, BackboneMixin):
#             module.gradient_checkpointing = value


class RELU(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, *args):
        assert len(args) == 1
        input_val = args[0]
        result = elementwise(FuncEnum.RELU)(input_val)
        return result


class SIGMOID(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, *args):
        assert len(args) == 1
        input_val = args[0]
        result = elementwise(FuncEnum.SIGMOID)(input_val)
        return result


class AITVitMatteBasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = nn.batch_norm.BatchNorm2d(
            out_channels, eps=config.batch_norm_eps
        )
        self.relu = RELU()

    def forward(self, hidden_state):
        hidden_state = ops.permute021()(ops.permute0213()(hidden_state))
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = ops.permute0213()(ops.permute021()(hidden_state))
        return hidden_state


class AITVitMatteConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(self, config):
        super().__init__()

        in_channels = config.backbone_config.num_channels
        out_channels = config.convstream_hidden_sizes

        self.convs = []  # nn.ModuleList()
        self.conv_chans = [in_channels] + out_channels

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(AITVitMatteBasicConv3x3(config, in_chan_, out_chan_))
        self.convs = nn.ModuleList(self.convs)

    def forward(self, pixel_values):
        # Start with the original pixel_values
        embeddings = ops.identity()(pixel_values)
        embeddings_list = (embeddings,)
        for i in range(len(self.convs)):
            embeddings = self.convs[i](embeddings)
            embeddings_list += (embeddings,)



        return embeddings_list


class AITVitMatteFusionBlock(nn.Module):
    """
    Simple fusion block to fuse features from ConvStream and Plain Vision Transformer.
    """

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.conv = AITVitMatteBasicConv3x3(
            config, in_channels, out_channels, stride=1, padding=1
        )
        # print(in_channels, out_channels)

    def forward(self, features, detailed_feature_map):
        features = ops.permute021()(ops.permute0213()(features))
        detailed_feature_map = ops.permute021()(ops.permute0213()(detailed_feature_map))
        upscaled_features = ops.upsampling2d(scale_factor=2, mode="bilinear")(features)
        out = ops.concatenate()([detailed_feature_map, upscaled_features], dim=3)
        # print(out.shape)
        # print(self.conv)
        out = ops.permute0213()(ops.permute021()(out))
        out = self.conv(out)

        return out


class AITVitMatteHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(self, config):
        super().__init__()

        in_channels = config.fusion_hidden_sizes[-1]
        mid_channels = 16

        self.matting_convs = nn.Sequential(
            nn.Conv2dBias(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.batch_norm.BatchNorm2d(mid_channels),
            RELU(),
            nn.Conv2dBias(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, hidden_state):
        hidden_state = ops.permute021()(ops.permute0213()(hidden_state))
        hidden_state = self.matting_convs(hidden_state)
        hidden_state = ops.permute0213()(ops.permute021()(hidden_state))

        return hidden_state


class AITVitMatteDetailCaptureModule(nn.Module):
    """
    Simple and lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(self, config):
        super().__init__()
        if len(config.fusion_hidden_sizes) != len(config.convstream_hidden_sizes) + 1:
            raise ValueError(
                "The length of fusion_hidden_sizes should be equal to the length of convstream_hidden_sizes + 1."
            )

        self.config = config
        self.convstream = AITVitMatteConvStream(config)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blocks = nn.ModuleList()
        self.fusion_channels = [config.hidden_size] + config.fusion_hidden_sizes

        for i in range(len(self.fusion_channels) - 1):
            self.fusion_blocks.append(
                AITVitMatteFusionBlock(
                    config=config,
                    in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                )
            )

        self.matting_head = AITVitMatteHead(config)
        self.sig = SIGMOID()

    def forward(self, features, pixel_values):
        detail_features = self.convstream(pixel_values)
        for i in range(len(self.fusion_blocks)):
            detailed_feature_map_name = len(self.fusion_blocks) - i - 1
            features = self.fusion_blocks[i](
                features, detail_features[detailed_feature_map_name]
            )

        alphas = self.sig(self.matting_head(features))

        return alphas


# VITMATTE_START_DOCSTRING = r"""
#     Parameters:
#     This model is a PyTorch [torch.tnn.Module](https://pytorch.org/docs/stable/tnn.html#torch.tnn.Module) sub-class. Use
#     it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
#     behavior.
#         config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
#             Initializing with a config file does not load the weights associated with the model, only the
#             configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# """

# VITMATTE_INPUTS_DOCSTRING = r"""
#     Args:
#         pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
#             Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
#             [`AutoImageProcessor`]. See [`AITVitMatteImageProcessor.__call__`] for details.
#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
#             `attentions` under returned tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
#             returned tensors for more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# """


# @add_start_docstrings(
#     """ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
#     VITMATTE_START_DOCSTRING,
# )
# class AITVitMatteForImageMatting(AITVitMattePreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config

#         print("backbone name", config.backbone_config)

#         self.backbone = AutoBackbone.from_config(config.backbone_config)
#         self.decoder = AITVitMatteDetailCaptureModule(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(VITMATTE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @replace_return_docstrings(output_type=ImageMattingOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         labels: Optional[torch.Tensor] = None,
#         return_dict: Optional[bool] = None,
#     ):
#         """
#         labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
#             Ground truth image matting for computing the loss.

#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AITVitMatteImageProcessor, AITVitMatteForImageMatting
#         >>> import torch
#         >>> from PIL import Image
#         >>> from huggingface_hub import hf_hub_download

#         >>> processor = AITVitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
#         >>> model = AITVitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

#         >>> filepath = hf_hub_download(
#         ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
#         ... )
#         >>> image = Image.open(filepath).convert("RGB")
#         >>> filepath = hf_hub_download(
#         ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
#         ... )
#         >>> trimap = Image.open(filepath).convert("L")

#         >>> # prepare image + trimap for the model
#         >>> inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

#         >>> with torch.no_grad():
#         ...     alphas = model(**inputs).alphas
#         >>> print(alphas.shape)
#         torch.Size([1, 1, 640, 960])
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

#         outputs = self.backbone.forward_with_filtered_kwargs(
#             pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
#         )

#         features = outputs.feature_maps[-1]
#         alphas = self.decoder(features, pixel_values)

#         # Save the tensors to files
#         # torch.save(features, 'features.pt')
#         # torch.save(pixel_values, 'pixel_values.pt')
#         # torch.save(alphas, 'alphas.pt')

#         # # Load the tensors back from the files
#         # loaded_features = torch.load('features.pt')
#         # loaded_pixel_values = torch.load('pixel_values.pt')

#         loss = None
#         if labels is not None:
#             raise NotImplementedError("Training is not yet supported")

#         if not return_dict:
#             output = (alphas,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return ImageMattingOutput(
#             loss=loss,
#             alphas=alphas,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
