import torch
import torch.nn as nn

from src.module.diffusionnet_pb import DiffusionNetBlock
from src.infra.registry import MODULE_REGISTRY

from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional

@MODULE_REGISTRY.register()
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        
        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        # self-attention block
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        # feed forward block
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

@MODULE_REGISTRY.register()
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # attention
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        
        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._ca_block(
                self.norm1(src), self.norm1(tgt), self.norm1(tgt), attn_mask, key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._ca_block(src, tgt, tgt, attn_mask, key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _ca_block(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        # self-attention block
        x = self.cross_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        # feed forward block
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


@MODULE_REGISTRY.register()
class DiffAttnFeat(nn.Module):
    def __init__(
            self,
            C_in, C_out, C_width=128,
            N_block=4, last_activation=None,
            mlp_hidden_dims=None, dropout=True, 
            with_gradient_features=True, with_gradient_rotations=True,
            diffusion_method='spectral',
            self_attn_layer_num=[1, 1, 0, 0], self_attn_head=4,
            cross_attn_layer_num=[0, 0, 1, 1], cross_attn_head=4,
        ):
        super().__init__()
        
        # --- settings ---
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.last_activation = last_activation

        # mlp settings
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # diffusion settings
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # attention settings
        self.self_attn_layer_num = self_attn_layer_num
        self.self_attn_head = self_attn_head
        self.cross_attn_layer_num = cross_attn_layer_num
        self.cross_attn_head = cross_attn_head

        # --- setup the network ---

        # first and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # blocks
        self.blocks = []

        for i_block in range(self.N_block):
            block = nn.ModuleDict({
                'diffusionnet': DiffusionNetBlock(
                    C_width = C_width,
                    mlp_hidden_dims = mlp_hidden_dims,
                    dropout = dropout,
                    diffusion_method = diffusion_method,
                    with_gradient_features = with_gradient_features,
                    with_gradient_rotations = with_gradient_rotations,
                ),
                'self_attn': nn.ModuleList([
                    SelfAttentionBlock(
                        d_model=C_width,
                        nhead=self.self_attn_head,
                        batch_first=True,
                    )
                    for _ in range(self.self_attn_layer_num[i_block])
                ]),
                'cross_attn': nn.ModuleList([
                    CrossAttentionBlock(
                        d_model=C_width,
                        nhead=self.cross_attn_head,
                        batch_first=True,
                    )
                    for _ in range(self.cross_attn_layer_num[i_block])
                ]),
            })
            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(
            self,
            descriptor_x, descriptor_y,
            verts_mask_x, verts_mask_y, 
            mass_x, mass_y,
            L_x=None, L_y=None,
            evals_x=None, evals_y=None,
            evecs_x=None, evecs_y=None,
            gradX_x=None, gradX_y=None,
            gradY_x=None, gradY_y=None,
        ):
        # Apply the first linear layer
        x = self.first_lin(descriptor_x) * verts_mask_x.unsqueeze(-1)
        y = self.first_lin(descriptor_y) * verts_mask_y.unsqueeze(-1)
      
        # Apply each of the blocks
        for block in self.blocks:
            x = block['diffusionnet'](
                x, mass_x, L_x, evals_x, evecs_x, gradX_x, gradY_x,
            )
            y = block['diffusionnet'](
                y, mass_y, L_y, evals_y, evecs_y, gradX_y, gradY_y,
            )

            for self_attn in block['self_attn']:
                x = self_attn(x, key_padding_mask=~verts_mask_x)
                y = self_attn(y, key_padding_mask=~verts_mask_y)

            for cross_attn in block['cross_attn']:
                x = cross_attn(x, y, key_padding_mask=~verts_mask_y)
                y = cross_attn(y, x, key_padding_mask=~verts_mask_x)

        # Apply the last linear layer
        x = self.last_lin(x)
        y = self.last_lin(y)

        return x * verts_mask_x.unsqueeze(-1), y * verts_mask_y.unsqueeze(-1) # mask out padded points [B, V, D] * [B, V, 1]
    

@MODULE_REGISTRY.register()
class AttentiveManifoldFeatureRefiner(nn.Module):
    def __init__(
            self,
            layer_str: str = 'sscs',
            feat_dim: int = 256,
            attn_head: int = 4,
            **kwargs,
        ):
        super().__init__()
        
        self.layer_str = layer_str
        self.feat_dim = feat_dim
        self.attn_head = attn_head

        self.layers = nn.ModuleList()
        for layer_type in layer_str:
            if layer_type == 's':
                self.layers.append(SelfAttentionBlock(
                    d_model=feat_dim, nhead=attn_head, **kwargs,
                ))

            elif layer_type == 'c':
                self.layers.append(CrossAttentionBlock(
                    d_model=feat_dim, nhead=attn_head, **kwargs,
                ))

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
    def forward(self, feat_x, feat_y, verts_mask_x, verts_mask_y):
        for layer in self.layers:
            if isinstance(layer, SelfAttentionBlock):
                feat_x = layer(feat_x, key_padding_mask=~verts_mask_x)
                feat_y = layer(feat_y, key_padding_mask=~verts_mask_y)

            elif isinstance(layer, CrossAttentionBlock):
                feat_x = layer(feat_x, feat_y, key_padding_mask=~verts_mask_y)
                feat_y = layer(feat_y, feat_x, key_padding_mask=~verts_mask_x)

            else:
                raise ValueError("Layer must be either SelfAttentionBlock or CrossAttentionBlock")
            
        return feat_x, feat_y