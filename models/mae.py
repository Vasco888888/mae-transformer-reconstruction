import torch
import torch.nn as nn

def patchify(imgs, p=16):
  """
  Logically divide the image into patches and flatten them
  Input: (n, 3, h, w)
  Output: (n, num_patches, 768)
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
  Input: (n, num_patches, 768)
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