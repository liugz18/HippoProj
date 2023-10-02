import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union

from transformers.models.vitdet.configuration_vitdet import VitDetConfig


from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.public import elementwise, FuncEnum
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code



config = {
  "drop_path_rate": 0.0,
  "dropout_prob": 0.0,
  "hidden_act": "gelu",
  "hidden_size": 384,
  "image_size": 512,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-06,
  "mlp_ratio": 4,
  "model_type": "vitdet",
  "num_attention_heads": 6,
  "num_channels": 4,
  "num_hidden_layers": 12,
  "out_features": [
    "stage12"
  ],
  "out_indices": [
    12
  ],
  "patch_size": 16,
  "pretrain_image_size": 224,
  "qkv_bias": True,
  "residual_block_indices": [
    2,
    5,
    8,
    11
  ],
  "stage_names": [
    "stem",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
    "stage8",
    "stage9",
    "stage10",
    "stage11",
    "stage12"
  ],
  "transformers_version": "4.34.0.dev0",
  "use_absolute_position_embeddings": True,
  "use_relative_position_embeddings": True,
  "window_block_indices": [
    0,
    1,
    3,
    4,
    6,
    7,
    9,
    10
  ],
  "window_size": 14
}
AITVitDetConfig = VitDetConfig(**config)

# class VitDetEmbeddings(nn.Module):
#     """
#     This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
#     `hidden_states` (patch embeddings) to be consumed by a Transformer.
#     """

#     def __init__(self, config):
#         super().__init__()
#         image_size, patch_size = config.pretrain_image_size, config.patch_size
#         num_channels, hidden_size = config.num_channels, config.hidden_size

#         image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
#         patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
#         num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_channels = num_channels
#         self.num_patches = num_patches

#         if config.use_absolute_position_embeddings:
#             # Initialize absolute positional embedding with pretrain image size.
#             num_positions = num_patches + 1
#             self.position_embeddings = nn.Parameter(shape=[1, num_positions, config.hidden_size])
#         else:
#             self.position_embeddings = None

#         self.projection = nn.Conv2dBias(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

#     def get_absolute_positions(self, abs_pos_embeddings, has_cls_token, height, width):
#         """
#         Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token dimension for the
#         original embeddings.

#         Args:
#             abs_pos_embeddings (`torch.Tensor`):
#                 Absolute positional embeddings with (1, num_position, num_channels).
#             has_cls_token (`bool`):
#                 If true, has 1 embedding in abs_pos_embeddings for cls token.
#             height (`int`):
#                 Height of input image tokens.
#             width (`int`):
#                 Width of input image tokens.

#         Returns:
#             Absolute positional embeddings after processing with shape (1, height, width, num_channels)
#         """
#         if has_cls_token:
#             abs_pos_embeddings = abs_pos_embeddings[:, 1:]
#         num_position = abs_pos_embeddings.shape[1]
#         size = int(math.sqrt(num_position))
#         if size * size != num_position:
#             raise ValueError("Absolute position embeddings must be a square number.")

#         if size != height or size != width:
#             new_abs_pos_embeddings = nn.functional.interpolate(
#                 abs_pos_embeddings.reshape(1, size, size, -1).permute(0, 3, 1, 2),
#                 size=(height, width),
#                 mode="bicubic",
#                 align_corners=False,
#             )

#             return new_abs_pos_embeddings.permute(0, 2, 3, 1)
#         else:
#             return abs_pos_embeddings.reshape(1, height, width, -1)

#     def forward(self, pixel_values: Tensor) -> Tensor:
#         num_channels = pixel_values.shape[1]
#         if num_channels != self.num_channels:
#             raise ValueError(
#                 "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
#                 f" Expected {self.num_channels} but got {num_channels}."
#             )
#         embeddings = self.projection(pixel_values)

#         if self.position_embeddings is not None:
#             # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
#             embeddings = embeddings.permute(0, 2, 3, 1)
#             # add position embeddings
#             embeddings = embeddings + self.get_absolute_positions(
#                 self.position_embeddings, True, embeddings.shape[1], embeddings.shape[2]
#             )
#             # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
#             embeddings = embeddings.permute(0, 3, 1, 2)

#         return embeddings


# def get_rel_pos(q_size, k_size, rel_pos):
#     """
#     Get relative positional embeddings according to the relative positions of query and key sizes.

#     Args:
#         q_size (`int`):
#             Size of query q.
#         k_size (`int`):
#             Size of key k.
#         rel_pos (`torch.Tensor`):
#             Relative position embeddings (num_embeddings, num_channels).

#     Returns:
#         Extracted positional embeddings according to relative positions.
#     """
#     max_rel_dist = int(2 * max(q_size, k_size) - 1)
#     # Interpolate rel pos if needed.
#     if rel_pos.shape[0] != max_rel_dist:
#         # Interpolate rel position embeddings.
#         rel_pos_resized = ops.upsampling2d(scale_factor=2, mode="bilinear")(
#             rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
#             size=max_rel_dist,
#             mode="linear",
#         )
#         rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
#     else:
#         rel_pos_resized = rel_pos

#     # Scale the coords with short length if shapes for q and k are different.
#     q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
#     k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
#     relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

#     return rel_pos_resized[relative_coords.long()]

class AITVitDetMlp(nn.Module):
    def __init__(self, config, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, specialization="fast_gelu")
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.dropout_prob if config else 0.1) 

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

class AITVitDetLayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over the
    channel dimension for inputs that have shape (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6, dtype="float16"):
        super().__init__()
        self.weight = nn.Parameter(shape=[normalized_shape], dtype=dtype)
        self.bias = nn.Parameter(shape=[normalized_shape], dtype=dtype)
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.dtype=dtype

    def forward(self, x):
        u = ops.reduce_mean(dim=3, keepdim=True)(x)
        s = ops.vector_norm(dim=3, keepdim=True)(x - u)
        s = s / math.sqrt(self.normalized_shape[0]) + self.eps
        x = (x - u) / s
        w = ops.unsqueeze(0)(ops.unsqueeze(0)(self.weight._tensor))
        # print(w.shape())
        x = w * x 
        x = x + ops.unsqueeze(0)(ops.unsqueeze(0)(self.bias._tensor))
        return x



class AITVitDetResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer. It contains 3 conv layers with kernels
    1x1, 3x3, 1x1.
    """

    def __init__(self, config, in_channels, out_channels, bottleneck_channels):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            in_channels (`int`):
                Number of input channels.
            out_channels (`int`):
                Number of output channels.
            bottleneck_channels (`int`):
                Number of output channels for the 3x3 "bottleneck" conv layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, 1)
        self.norm1 = AITVitDetLayerNorm(bottleneck_channels)
        self.act1 = nn.activation.GELU()

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, padding=1)
        self.norm2 = AITVitDetLayerNorm(bottleneck_channels)
        self.act2 = nn.activation.GELU()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, 1)
        self.norm3 = AITVitDetLayerNorm(out_channels)

    def forward(self, x):
        out = x
        out = ops.permute021()(ops.permute0213()(out)) # permute (B, C, H, W) -> (B, H, W, C)
        for layer in self.children():
            out = layer(out)

        out = ops.permute0213()(ops.permute021()(out)) # permute (B, H, W, C) -> (B, C, H, W)
        out = x + out
        return out


class AITVitDetAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, input_size=None):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            input_size (`Tuple[int]`, *optional*):
                Input resolution, only required in case relative position embeddings are added.
        """
        super().__init__()

        dim = config.hidden_size
        num_heads = config.num_attention_heads

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_relative_position_embeddings = config.use_relative_position_embeddings
        if self.use_relative_position_embeddings:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(shape=[2 * input_size[0] - 1, head_dim])
            self.rel_pos_w = nn.Parameter(shape=[2 * input_size[1] - 1, head_dim])

    def forward(self, hidden_state):
        batch_size, height, width, _ = hidden_state.shape()
        # qkv with shape (3, batch_size, num_heads, height * width, num_channels)
        qkv = self.qkv(hidden_state)
        qkv = ops.permute()(ops.reshape()(qkv, [batch_size, height * width, 3, self.num_heads, -1]), [2, 0, 3, 1, 4])
        # queries, keys and values have shape (batch_size * num_heads, height * width, num_channels)
        qkv = ops.reshape()(qkv, [3, batch_size * self.num_heads, height * width, -1])
        queries, keys, values = ops.split()(qkv, 1, dim=0)#[0], qkv[1], qkv[2]

        queries, keys, values = ops.squeeze(0)(queries), ops.squeeze(0)(keys), ops.squeeze(0)(values)
        keys = ops.permute021()(keys)
        qk = ops.bmm_rrr()(queries, keys)
        score = ops.elementwise(FuncEnum.MUL)(qk, self.scale)
        score = ops.softmax()(score, -1)
        hidden_state = ops.bmm_rrr()(score, values)

        # hidden_state = attention_probs @ values
        hidden_state = ops.reshape()(hidden_state, [batch_size, self.num_heads, height, width,  -1])#(batch_size, self.num_heads, height, width, -1)
        hidden_state = ops.permute()(hidden_state, [0, 2, 3, 1, 4])
        hidden_state = ops.reshape()(hidden_state, [batch_size, height, width, -1])
        hidden_state = self.proj(hidden_state)

        outputs = (hidden_state,)

        return outputs


def window_partition(hidden_state, window_size):
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        hidden_state (`torch.Tensor`):
            Input tokens with [batch_size, height, width, num_channels].
        window_size (`int`):
            Window size.

    Returns:
        `tuple(torch.FloatTensor)` comprising various elements:
        - windows: windows after partition with [batch_size * num_windows, window_size, window_size, num_channels].
        - (patch_height, patch_width): padded height and width before partition
    """
    batch_size, height, width, num_channels = hidden_state.shape

    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    patch_height, patch_width = height + pad_height, width + pad_width
    if pad_height > 0 or pad_width > 0:
        hidden_state = ops.expand()(hidden_state, (batch_size, patch_height, patch_width))
    

    hidden_state = ops.reshape()(hidden_state,[
        batch_size, patch_height // window_size, window_size, patch_width // window_size, window_size, num_channels]
    )
    windows = ops.reshape()(ops.permute()(hidden_state, [0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, num_channels])
    return windows, (patch_height, patch_width)

def window_unpartition(windows, window_size, pad_height_width, height_width):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (`torch.Tensor`):
            Input tokens with [batch_size * num_windows, window_size, window_size, num_channels].
        window_size (`int`):
            Window size.
        pad_height_width (`Tuple[int]`):
            Padded height and width (patch_height, patch_width).
        height_width (`Tuple[int]`):
            Original height and width before padding.

    Returns:
        hidden_state: unpartitioned sequences with [batch_size, height, width, num_channels].
    """
    patch_height, patch_width = pad_height_width
    height, width = height_width
    batch_size = windows.shape[0] // (patch_height * patch_width // window_size // window_size)
    hidden_state = windows.view(
        batch_size, patch_height // window_size, patch_width // window_size, window_size, window_size, -1
    )
    hidden_state = ops.reshape(ops.permute(hidden_state, [0, 1, 3, 2, 4, 5]), [batch_size, patch_height, patch_width, -1])

    if patch_height > height or patch_width > width:
        # hidden_state = hidden_state[:, :height, :width, :].contiguous()
        hidden_state = ops.dynamic_slice()(
                hidden_state,
                start_indices=[0, 0, 0, 0],
                end_indices=[None, height, width, None],
            )
    return hidden_state

class AITVitDetLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self, config: VitDetConfig, drop_path_rate: float = 0, window_size: int = 0, use_residual_block: bool = False
    ) -> None:
        super().__init__()

        dim = config.hidden_size
        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)

        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = AITVitDetAttention(
            config, input_size=input_size if window_size == 0 else (window_size, window_size)
        )

        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = AITVitDetMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if self.use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = AITVitDetResBottleneckBlock(
                config=config,
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
            )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        hidden_states = ops.permute()(hidden_states, [0, 2, 3, 1])

        shortcut = hidden_states

        hidden_states = self.norm1(hidden_states)

        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)

        self_attention_outputs = self.attention(
            hidden_states
        )
        hidden_states = self_attention_outputs[0]

        # Reverse window partition
        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))

        # first residual connection
        hidden_states = shortcut + hidden_states

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        hidden_states = ops.permute()(hidden_states, [0, 3, 1, 2])

        if self.use_residual_block:
            hidden_states = self.residual(hidden_states)

        outputs = (hidden_states,)

        return outputs