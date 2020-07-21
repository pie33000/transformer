import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nodes.utils import clones, NormLayer, SubLayer

class Encoder(nn.Module):
    """
    Encoder
    """
    
    def __init__(self, layer, N):
        """
        Encoder constructor, stacks N encoder
        Args:
            layer: (nn.Module) MultiHeadAttention, FeedForward
            N: number of staked layer
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = NormLayer(layer.size)
    
    def forward(self, x, mask):
        """
        Forward pass, pass in all stacked layers
        Args:
            x: (torch.Tensor) input
        Return:
        
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Encoder Layer constructor
        Args:
            size: (integer) size of input (vector to represent a word)
            self_attn: (MultiHeadAttentionLayer)
            feed_forward: (FeedForward)
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.sublayer = clones(SubLayer(size, dropout), 2)
    
    def forward(self, x, mask):
        """
        Forward pass, pass in MultiHeadAttention (First sublayer) then in second (FeedForward layer)
        Args:
            x: (torch.Tensor) input
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)