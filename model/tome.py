# Note: This file is based on https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from model.vision_transformer_timm import Attention, Block, VisionTransformer

from model.adaptive_merge import bipartite_soft_matching, merge_source, merge_wavg


def get_merging_schedule(model_name, schedule_name):
    schedule = {
        "vit_base_patch16_224_in21k":{
            "low":[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 10],
            "high":[40, 34, 30, 24, 18, 14, 10, 8, 4, 4, 3, 3],
        },
        "vit_large_patch16_224_in21k":{
            "low":[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6],
            "high":[20, 19, 18, 17, 15, 13, 13, 12, 10, 9, 8, 6, 6, 4, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1],
        },
        "deit_base_distilled_patch16_224":{
            "low":[16, 16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 10],
            "high":[40, 34, 30, 24, 18, 14, 10, 8, 4, 4, 3, 3],
        },
    }

    return schedule[model_name][schedule_name]


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.visual_prompt_dim > 0:
            visual_prompt_tokens = self.visual_prompt_token.expand(B,-1,-1)

            visual_prompt_tokens = self.drop_prompt(visual_prompt_tokens)
            if self.last_prompt_dim == 0:
                x = torch.cat((x,visual_prompt_tokens), dim=1)
            else:
                x = torch.cat((x[:,:-self.last_prompt_dim,:],visual_prompt_tokens), dim=1)

        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"], 
                                                    pyra_weight=self.pyra,
                                                    is_training=self.training)

        x = x + self.adapter(self.drop_path(self.mlp(self.norm2(x))))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        if self.LoRA_dim > 0:
            qkv_delta = self.LoRA_a(self.LoRA_drop(x))
            qkv_delta = self.LoRA_b(qkv_delta).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_delta, k_delta, v_delta = qkv_delta.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            q,k,v = q+q_delta,k+k_delta,v+v_delta

        if self.prefix_dim > 0:
            prefix_tokens_key = self.prefix_tokens_key.expand(B,-1,-1)
            prefix_tokens_value = self.prefix_tokens_value.expand(B,-1,-1)
            k,v = torch.cat((k,prefix_tokens_key), dim=1), torch.cat((v,prefix_tokens_value), dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_tome(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
