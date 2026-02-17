import torch
import torch.nn as nn

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