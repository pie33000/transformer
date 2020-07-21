import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nodes.utils import clones, NormLayer, SubLayer

class Decoder(nn.Module):
    
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = NormLayer(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder Layer
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Decoder constructor
        Args:
            size: (integer)
            self_attn: (MultiHeadAttentionLayer)
            src_attn: (torch.tensor) comes from encoder output
            feed_forward: (FeedForward)
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.src_attn = src_attn
        self.size = size
        self.sublayer = clones(SubLayer(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Forward pass
        Args:
            x: (torch.tensor) input
            memory: (torch.tensor) key 
            src_mask: (torch.tensor) comes from encoder
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)