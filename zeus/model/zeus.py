from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from zeus.configs import LossType
from zeus.model.layer import TransformerEncoderLayer
from zeus.model.model_utils import SeqBN, bool_mask_to_att_mask
from sklearn.preprocessing import MinMaxScaler


class ZeusTransformerModel(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers,
                 dropout=0.0, *, n_clusters=10, dist_based_logit=False,
                 loss_type=LossType.CENTER, decoder=None,
                 input_normalization=False, pre_norm=False,
                 activation='gelu', recompute_attn=False, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.distance_based_logit = dist_based_logit
        self.loss_type = loss_type

        if not self.distance_based_logit:
            self.decoder = decoder(ninp, nhid, n_out) if decoder is not None \
                else nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        else:
            self.decoder = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU())
            self.out_layer = nn.Linear(nhid, n_out, bias=False)

        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.efficient_eval_masking = efficient_eval_masking
        self.full_attention = full_attention

        self.n_out = n_out
        self.nhid = nhid

        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, 1, ninp))

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, x, *, k=0):
        x_src = self.encoder(x)
        src_mask = None

        if src_mask is None:
            full_len = len(x_src) + len(self.cluster_centers)
            if self.full_attention:
                src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(x_src.device)
            elif self.efficient_eval_masking:
                src_mask = full_len
            else:
                src_mask = self.generate_D_q_matrix(full_len, 0).to(x_src.device)

        src = torch.cat([x_src, self.cluster_centers], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask)
        if self.loss_type == LossType.CENTER:
            return output

        #output = self.decoder(output)
#
        #if self.distance_based_logit:
        #    output = -torch.linalg.norm(output.unsqueeze(-1) - self.out_layer.weight.T, dim=-2)**2

        return output

    def predict_embedding(self, x, use_torch=True):
        x = x.unsqueeze(1)
        x = self.forward(x)
        x = x.squeeze(1)

        centers = len(self.cluster_centers)
        x = x[:-centers]

        if use_torch:
            x_min = torch.min(x, dim=-1, keepdim=True).values
            x_max = torch.max(x, dim=-1, keepdim=True).values
            x = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            scaler = MinMaxScaler((-1, 1))
            x = scaler.fit_transform(x.detach().cpu())
            x = torch.tensor(x, dtype=torch.float32)
        return x


class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
