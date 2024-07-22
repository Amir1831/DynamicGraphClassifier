import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config
class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )



    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x




class TemporalAttention(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = AttentionBlock(num_heads = num_heads, embed_dim = input_dim, hidden_dim = hidden_dim)
        self.Conv1 = nn.Conv2d(Config.P , Config.K_E , kernel_size=(3,3),padding=1)
        self.Conv2 = nn.Conv2d(Config.K_E , Config.K_E , kernel_size=(3,3),padding=1)

    def forward(self, x):
        # Input : (B , T , P , V)
        x = x.view(x.size(0) , x.size(2) , x.size(1) , x.size(3))  # Reshape to : (B , P , T , V)
        ## Apply Conv2D 
        x = self.Conv2(self.Conv1(x)) # Output of Conv : (B , K_E , T , V)
        x = x.view(x.size(0) , x.size(2) , x.size(1) , x.size(3))  # Change back dimention to : (B , T , K_E , V)
        ## Temporal Part 
        x = x.view(x.size(1), x.size(0), -1)  # merge spatial dims & Change dimention because "Batch_First" is Flase
        x = self.attention(x)
        x = x.view(x.size(1), x.size(0), x.size(2) // Config.V, x.size(2) // Config.K_E)  # restore spatial dims
        return x

class SpatialAttention(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.attention = AttentionBlock(num_heads = num_heads, embed_dim = input_dim, hidden_dim = hidden_dim)

    def forward(self, x):
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(1))  # merge temporal dim
        x = self.attention(x)
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(1))  # restore temporal dim
        return x

class CombinedAttention(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim):
        super(CombinedAttention, self).__init__()
        self.temporal_attention = TemporalAttention(num_heads, input_dim, hidden_dim)
        self.spatial_attention = SpatialAttention(num_heads, input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        temporal_output = self.temporal_attention(x)
        spatial_output = self.spatial_attention(x)
        output = torch.cat((temporal_output, spatial_output), dim=-1)
        output = self.linear(output)
        return output