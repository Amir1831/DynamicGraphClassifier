import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import CONFIG
Config = CONFIG()
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
        # Input : (B , T , V , P)
        x = x.transpose(1,3)  # Reshape to : (B , P , V , T)
        ## Apply Conv2D 
        x = self.Conv2(self.Conv1(x)) # Output of Conv : (B , K_E , V , T)
        x = x.transpose(1,3)  # Change back dimention to : (B , T , V , K_E)
        ## Temporal Part 
        x = x.reshape(x.size(1), x.size(0), -1)  # merge spatial dims & Change dimention because "Batch_First" is Flase
        x = self.attention(x)
        x = x.reshape(x.size(1), x.size(0), x.size(2) // Config.K_E ,  x.size(2) // Config.V)  # restore spatial dims
        return x

class DynamicMatrix(nn.Module):
    def __init__(self):
        """
        ## Dynamic adjacency matrix 
        To capture spatial relationships between brain regions, while considering the independently learned embeddings
        from the feature extractor, we employ a self-attention mechanism. Specifically, at each snapshot we utilize the em-
        beddings to learn the pair-wise dependency structure between brain regions, using a simplified version
        of scaled dot-product self-attention.
        ## Edge sparsity 
        By definition, the adjacency matrix represents a fully-connected graph at every snapshot. However, fully-connected graphs are challenging to
        interpret and computationally expensive for learning downstream tasks with GNNs. Moreover, they are susceptible to noise. To address these issues, we propose a soft threshold
        operator to enforce edge sparsity.This operator is defined following fθP (ai,j,t) = ReLU(ai,j,t − Sigmoid(θP)).
        """
        super().__init__()
        # initial Query and Key Parameters
        self.W_Q = nn.Parameter(torch.rand(Config.K_E , Config.K_S))
        self.W_K = nn.Parameter(torch.rand(Config.K_E , Config.K_S))
        # Initisl Theta_P for edge sparsity
        self.theta = nn.Parameter(torch.ones(1 , Config.T , Config.V , Config.V) * -10)  # intial theta with -10

        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)
    def forward(self , x):
        # INPUT : (B , T , V, K_E)
        B , _, _ , _ = x.shape
        theta = self.theta.repeat(B, 1 ,1, 1)
        Q_T = x @ self.W_Q  # (B , T , V , K_S)
        K_T = x @ self.W_K  # (B , T , V , K_S)
        return self.ReLU(self.Softmax((Q_T @ K_T.transpose(2,3)) / torch.sqrt(torch.tensor(Config.K_S))) - self.Softmax(theta))

class SpatialAttention(nn.Module):
    def __init__(self):
    
        super().__init__()
        # initial Query and Key Parameters
        self.W_Q = nn.Parameter(torch.rand(Config.K_E , Config.K_S))
        self.W_K = nn.Parameter(torch.rand(Config.K_E , Config.K_S))
        self.W_V = nn.Parameter(torch.rand(Config.K_E , Config.K_E))
        self.Softmax = nn.Softmax(dim=1)
    def forward(self , x):
        # INPUT : (B , T , V, K_E)
        Q_T = x @ self.W_Q  # (B , T , V , K_S)
        K_T = x @ self.W_K  # (B , T , V , K_S)
        V_T = x @ self.W_V  # (B , T , V , K_E)
        return self.Softmax((Q_T @ K_T.transpose(2,3)) / torch.sqrt(torch.tensor(Config.K_S))) @ V_T
