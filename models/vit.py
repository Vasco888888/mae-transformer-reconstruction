import torch
import torch.nn as nn

class Attention(nn.Module):
  """
  Multi-head self-attention module.
  """
  def __init__(self, dim, num_heads=8):
    super().__init__()
    self.num_heads = num_heads
    self.scale = (dim // num_heads) ** -0.5

    # Linear projections
    self.qkv = nn.Linear(dim, dim * 3, bias=False)
    self.proj = nn.Linear(dim, dim)
  
  def forward(self, x):
    B, L, C = x.shape

    # Split into Q, K, V
    qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Dot product attention: (Q @ K^T) / V
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)

    x = (attn @ v).transpose(1, 2).reshape(B, L, C)
    x = self.proj(x)
    return x

class Block(nn.Module):
  """
  A standard Transformer Block with Attention and a Multi-Layer Perceptron
  """
  def __init__(self, dim, num_heads):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn = Attention(dim, num_heads)
    self.norm2 = nn.LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, 4 * dim),
      nn.GELU(),
      nn.Linear(4 * dim, dim)
    )

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x