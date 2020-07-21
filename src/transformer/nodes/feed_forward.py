import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Norm Layer, follow the implementation in Layer Normalization bu Jimmy Lei Ba and al.
    (https://arxiv.org/pdf/1607.06450.pdf)
    """
    def __init__(self, d_model=512, inner_layer=2048, dropout=0.1):
        """
        Constructor of Normalization Layer
        
        Args:
            d_model: the size of the model (i.e size of vector which represents a word or sentence) (integer)
            inner_layer: the size of hidden layer between two linear transformations (integer)
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, inner_layer, bias=True)
        self.fc2 = nn.Linear(inner_layer, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for layers.
        
        Args:
            x: The input a vector of size d_model
        
        Return:
            A vector of size d_model
        
        """
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x