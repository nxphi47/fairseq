import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)

from .transformer import *


@register_model('dptree2seq')
class DPTree2SeqTransformer(TransformerModel):

    @classmethod
    def build_model(cls, args, task):
        return super().build_model(args, task)

    def forward(self, src_tokens, src_indices, src_lengths, prev_output_tokens):

        encoder_out = self.encoder(src_tokens, src_indices, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


@register_model_architecture('dptree2seq', 'dptree2seq')
def dptree2seq_base(args):
    base_architecture(args)


@register_model_architecture('dptree2seq', 'dptree2seq_tiny')
def dptree2seq_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)

    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de')
def dptree2seq_base_wmt_en_de(args):
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de_t2t')
def dptree2seq_base_wmt_en_de(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de_vas_big')
def dptree2seq_big_wmt_en_de_vas(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de_t2t_big')
def dptree2seq_big_wmt_en_de_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)

    dptree2seq_big_wmt_en_de_vas(args)






