import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

from .multihead_attention import *


class DPTreeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        # if self.in_proj_bias is not None:
        #     nn.init.constant_(self.in_proj_bias, 0.)
        #     nn.init.constant_(self.out_proj.bias, 0.)
        # if self.bias_k is not None:
        #     nn.init.xavier_normal_(self.bias_k)
        # if self.bias_v is not None:
        #     nn.init.xavier_normal_(self.bias_v)
        raise NotImplementedError

    def forward(self, query, key, value, indices, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.

        :param query:   [Tq, B, C]
        :param key:     [Tk, B, C]
        :param value:   [Tk, B, C]
        :param indices: [Tk, B, 2]
        :param key_padding_mask:    [B, Tk]
        """
        # TODO: to do in multihead_attention.py
        raise NotImplementedError




