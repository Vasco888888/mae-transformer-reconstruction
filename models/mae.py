import torch
import torch.nn as nn

def patchify(imgs, p=16):
  """
  Input: (n, 3, h, w)
  Output: (n, num_patches, 768)
  """

  n, c, h, w = imgs.shape
  h_p = h // p
  w_p = w // p

  x = imgs.reshape(shape=(n, 3, h_p, p, w_p, p))
  x  = x.permute(0, 2, 4, 3, 5, 1)
  x = x.reshape(shape=(n, h_p * w_p, p * p * 3))
  return x