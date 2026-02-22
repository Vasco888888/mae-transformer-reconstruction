import torch
import torch.nn as nn
from .vit import Block
from utils.pos_embed import get_2d_sincos_pos_embed

def patchify(imgs, p=16):
  """
  Logically divide the image into patches and flatten them
  Input: (n, 3, h, w)
  Output: (n, num_patches, dim)
  """

  n, c, h, w = imgs.shape
  h_p = h // p
  w_p = w // p

  # Reshape to (Batch, Channels, Num_patches_h, patch_h, Num_patches_w, patch_w)
  x = imgs.reshape(shape=(n, 3, h_p, p, w_p, p))

  # Move dimensions to group pixels of the same patch together
  x  = x.permute(0, 2, 4, 3, 5, 1)

  # Flatten to (Batch, Num_patches, Patch_pixels)
  x = x.reshape(shape=(n, h_p * w_p, p * p * 3))
  return x

def unpatchify(x, p=16):
  """
  Reconstruct the image from patches
  Input: (n, num_patches, dim)
  Output: (n, 3, h, w)
  """

  h_p = w_p = int(x.shape[1]**0.5)
  n = x.shape[0]

  # Reshape back to the 6D tensor
  x = x.reshape(shape=(n, h_p, w_p, p, p, 3))

  # Move Channels back to the second dimension
  x = x.permute(0, 5, 1, 3, 2, 4)

  # Merge patches back into an image
  imgs = x.reshape(shape=(n, 3, h_p * p, w_p * p))
  return imgs

def random_masking(x, mask_ratio=0.75):
  """
  Randomly hides patches based on mask_ratio
  Input: (n, num_patches, dim)
  Output: (n, num_patches_kept, dim), mask, ids_restore
  """

  n, num_patches, dim = x.shape # Batch, Num_patches, Dim
  len_keep = int(num_patches * (1 - mask_ratio))

  # Generate random noise for each patch
  noise = torch.rand((n, num_patches), device=x.device)

  # Sort noise to get random indices
  ids_shuffle = torch.argsort(noise, dim=1)
  ids_restore = torch.argsort(ids_shuffle, dim=1)

  # Select the 25% of patches to keep
  ids_keep = ids_shuffle[:, :len_keep]
  x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

  # Generate a binary mask for visualization (0=keep, 1=remove)
  mask = torch.ones((n, num_patches), device=x.device)
  mask[:, :len_keep] = 0
  mask = torch.gather(mask, dim=1, index=ids_restore)

  return x_masked, mask, ids_restore

class MaskedAutoencoderViT(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3,
              embed_dim=768, depth=12, num_heads=12,
              decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
              mlp_ratio=4., norm_layer=nn.LayerNorm):
    super().__init__()

    # -------------------------------------
    # MAE Encoder
    
    self.patch_embed = nn.Linear(patch_size ** 2 * in_chans, embed_dim)
    num_patches = (img_size // patch_size) ** 2
    
    # Positional encoding
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

    self.blocks = nn.ModuleList([
        Block(embed_dim, num_heads) for _ in range(depth)])

    self.norm = norm_layer(embed_dim)

    # -------------------------------------
    # MAE Decoder
    
    self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
    self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

    self.decoder_blocks = nn.ModuleList([
      Block(decoder_embed_dim, decoder_num_heads) for i in range(decoder_depth)])

    self.decoder_norm = norm_layer(decoder_embed_dim)
    self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)    
    # -------------------------------------

    self.patch_size = patch_size
    self.initialize_weights()

  def unpatchify(self, x):
    return unpatchify(x, self.patch_size)

  def patchify(self, imgs):
    return patchify(imgs, self.patch_size)

  def initialize_weights(self):
    # Initialize positional encodings
    pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[1]**.5))
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.decoder_pos_embed.shape[1]**.5))
    self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
  
  def forward_encoder(self, x, mask_ratio):
    # Flatten image to patches
    x = self.patchify(x)
    # Add positional encoding
    x = x + self.pos_embed
    # Mask 75% of the data
    x, mask, ids_restore = random_masking(x, mask_ratio)
    # Run through Transformer blocks
    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)
    return x, mask, ids_restore
  
  def forward_decoder(self, x, ids_restore):
    # Project encoder output to decoder dimension
    x = self.decoder_embed(x)

    # Append mask tokens to the sequence
    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    x_all = torch.cat([x, mask_tokens], dim=1)
    
    # Unshuffle: put visible and mask tokens in their original 2D positions
    x = torch.gather(x_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

    # Add decoder positional encoding
    x = x + self.decoder_pos_embed
    
    for blk in self.decoder_blocks:
      x = blk(x)
    x = self.decoder_norm(x)
    
    # Predict pixel values
    x = self.decoder_pred(x)
    return x

  def forward(self, imgs, mask_ratio=0.75):
    latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    pred = self.forward_decoder(latent, ids_restore)
    return pred, mask

  def forward_loss(self, imgs, pred, mask):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = self.patchify(imgs)
    
    # Mean Squared Error
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1) # Loss per patch

    # Mean loss on masked patches only
    loss = (loss * mask).sum() / mask.sum()  

    return loss
