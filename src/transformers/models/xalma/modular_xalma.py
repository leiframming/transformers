# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""X-ALMA model implementation using modular transformers."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
)
from .configuration_xalma import XALMAConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "XALMAConfig"

__all__ = [
    "XALMAForCausalLM",
    "XALMAModel",
    "XALMAPreTrainedModel",
    "XALMAForSequenceClassification",
    "XALMAForQuestionAnswering",
    "XALMAForTokenClassification",
]

LANG_TABLE = {
    "en": "English",
    # Group 1:
    "da": "Danish",
    "nl": "Dutch",
    "de": "German",
    "is": "Icelandic",
    "no": "Norwegian",
    "sv": "Swedish",
    "af": "Afrikaans",
    # Group 2:
    "ca": "Catalan",
    "ro": "Romanian",
    "gl": "Galician",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    # Group 3:
    "bg": "Bulgarian",
    "mk": "Macedonian",
    "sr": "Serbian",
    "uk": "Ukrainian",
    "ru": "Russian",
    # Group 4:
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "mg": "Malagasy",
    "fr": "French",
    # Group 5:
    "hu": "Hungarian",
    "el": "Greek",
    "cs": "Czech",
    "pl": "Polish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    # Group 6:
    "ka": "Georgian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fi": "Finnish",
    "et": "Estonian",
    # Group 7:
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "ur": "Urdu",
    # Group 8:
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tr": "Turkish",
    "uz": "Uzbek",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
}

GROUP2LANG = {
    1: ["da", "nl", "de", "is", "no", "sv", "af"],
    2: ["ca", "ro", "gl", "it", "pt", "es"],
    3: ["bg", "mk", "sr", "uk", "ru"],
    4: ["id", "ms", "th", "vi", "mg", "fr"],
    5: ["hu", "el", "cs", "pl", "lt", "lv"],
    6: ["ka", "zh", "ja", "ko", "fi", "et"],
    7: ["gu", "hi", "mr", "ne", "ur"],
    8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
}

LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}


# X-ALMA MLP extends Llama MLP by adding language-specific LoRA modules
class XALMAMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.lora_size = config.lora_size
        self.lora_alpha = config.lora_alpha

        self.gate_lora_A = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_size, self.lora_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )
        self.gate_lora_B = nn.ModuleDict(
            {
                str(i): nn.Linear(self.lora_size, self.intermediate_size, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.up_lora_A = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_size, self.lora_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )
        self.up_lora_B = nn.ModuleDict(
            {
                str(i): nn.Linear(self.lora_size, self.intermediate_size, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.down_lora_A = nn.ModuleDict(
            {
                str(i): nn.Linear(self.intermediate_size, self.lora_size, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.down_lora_B = nn.ModuleDict(
            {str(i): nn.Linear(self.lora_size, self.hidden_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )

    def forward(self, x, lang=""):
        gate_proj_weight = (
            self.gate_proj.weight
            + self.gate_lora_B[LANG2GROUP[lang]].weight @ self.gate_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )
        up_proj_weight = (
            self.up_proj.weight
            + self.up_lora_B[LANG2GROUP[lang]].weight @ self.up_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )
        down_proj_weight = (
            self.down_proj.weight
            + self.down_lora_B[LANG2GROUP[lang]].weight @ self.down_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )

        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = gate_proj_weight.split(slice, dim=0)
            up_proj_slices = up_proj_weight.split(slice, dim=0)
            down_proj_slices = down_proj_weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            x = self.act_fn(F.linear(x, gate_proj_weight)) * F.linear(x, up_proj_weight)
            down_proj = F.linear(x, down_proj_weight)
        return down_proj


# X-ALMA Attention extends Llama Attention by adding language-specific LoRA modules
class XALMAAttention(LlamaAttention):
    def __init__(self, config: XALMAConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.lora_size = config.lora_size
        self.lora_alpha = config.lora_alpha

        self.q_lora_A = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_size, self.lora_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )
        self.q_lora_B = nn.ModuleDict(
            {
                str(i): nn.Linear(self.lora_size, self.num_heads * self.head_dim, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.k_lora_A = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_size, self.lora_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )
        self.k_lora_B = nn.ModuleDict(
            {
                str(i): nn.Linear(self.lora_size, self.num_key_value_heads * self.head_dim, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.v_lora_A = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_size, self.lora_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )
        self.v_lora_B = nn.ModuleDict(
            {
                str(i): nn.Linear(self.lora_size, self.num_key_value_heads * self.head_dim, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.o_lora_A = nn.ModuleDict(
            {
                str(i): nn.Linear(self.num_heads * self.head_dim, self.lora_size, bias=False)
                for i in range(1, len(GROUP2LANG) + 1)
            }
        )
        self.o_lora_B = nn.ModuleDict(
            {str(i): nn.Linear(self.lora_size, self.hidden_size, bias=False) for i in range(1, len(GROUP2LANG) + 1)}
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        lang: str = "",
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_proj_weight = (
            self.q_proj.weight
            + self.q_lora_B[LANG2GROUP[lang]].weight @ self.q_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )
        k_proj_weight = (
            self.k_proj.weight
            + self.k_lora_B[LANG2GROUP[lang]].weight @ self.k_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )
        v_proj_weight = (
            self.v_proj.weight
            + self.v_lora_B[LANG2GROUP[lang]].weight @ self.v_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )
        o_proj_weight = (
            self.o_proj.weight
            + self.o_lora_B[LANG2GROUP[lang]].weight @ self.o_lora_A[LANG2GROUP[lang]].weight * self.lora_alpha
        )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = q_proj_weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = k_proj_weight.split(key_value_slicing, dim=0)
            value_slices = v_proj_weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = F.linear(hidden_states, q_proj_weight)
            key_states = F.linear(hidden_states, k_proj_weight)
            value_states = F.linear(hidden_states, v_proj_weight)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = o_proj_weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = F.linear(attn_output, o_proj_weight)

        return attn_output, None


# X-ALMA DecoderLayer extends Llama DecoderLayer to use XALMA components
class XALMADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: XALMAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = XALMAAttention(config=config, layer_idx=layer_idx)
        self.mlp = XALMAMLP(config)


class XALMAPreTrainedModel(LlamaPreTrainedModel):
    config_class = XALMAConfig


class XALMAModel(LlamaModel):
    pass


class XALMAForCausalLM(LlamaForCausalLM):
    pass


class XALMAForSequenceClassification(LlamaForSequenceClassification):
    pass


class XALMAForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class XALMAForTokenClassification(LlamaForTokenClassification):
    pass
