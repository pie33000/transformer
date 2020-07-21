import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nodes.utils import clones
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    """
    Multi Head Attetion Network
    """
    def __init__(self, h=8, d_model=512, dropout=0.1):
        """
        Constructor for Encoder
        
        Args:
            d_model:
            h: number of parallel attention layers
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for layers.
        
        Args:
            x: The input a vector of size d_model
        
        Return:
            A vector of size d_model
        
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def attention(query, key, value, mask=None, dropout=None):
    """
    Attention function compute scaled dot product attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn