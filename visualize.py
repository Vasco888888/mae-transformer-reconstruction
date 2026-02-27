import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.mae import MaskedAutoencoderViT
from utils.datasets import build_dataset
from train import Args

def show_image(image, title=''):
  # Inverse normalize
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  image = image.cpu() * std + mean
  image = torch.clamp(image, 0, 1)
    
  plt.imshow(image.permute(1, 2, 0).numpy())
  plt.title(title, fontsize=16)
  plt.axis('off')

def visualize(image_path=None, mask_ratio=None):
  args = Args()
  args.train.device = "cpu"
  
  final_mask_ratio = mask_ratio if mask_ratio is not None else args.train.mask_ratio
    
  if image_path:
    actual_path = image_path
    if not os.path.exists(actual_path) and hasattr(args.data, 'assets_dir'):
      candidate_path = os.path.join(args.data.assets_dir, image_path)
      if os.path.exists(candidate_path):
        actual_path = candidate_path

    img_pil = Image.open(actual_path).convert('RGB')
    
    preprocess = transforms.Compose([
      transforms.Resize(32),        
      transforms.Resize(224),       
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = preprocess(img_pil).unsqueeze(0).to(args.train.device)
  else:
    dataset = build_dataset(args, is_train=False)
    img, _ = dataset[0]  
    img = img.unsqueeze(0).to(args.train.device)

  model = MaskedAutoencoderViT(
    patch_size=args.model.patch_size, 
    embed_dim=args.model.embed_dim, 
    depth=args.model.depth, 
    num_heads=args.model.num_heads,
    decoder_embed_dim=args.model.decoder_embed_dim, 
    decoder_depth=args.model.decoder_depth, 
    decoder_num_heads=args.model.decoder_num_heads,
  ).to(args.train.device)

  model.load_state_dict(torch.load("checkpoints/mae_checkpoint_epoch_50.pth", map_location="cpu"))
  model.eval()

  with torch.no_grad():
    pred, mask = model(img, mask_ratio=final_mask_ratio)
    rec_img = model.unpatchify(pred)
        
    p = model.patch_size
    h = w = img.shape[2] // p
    mask_expanded = mask.reshape(img.shape[0], 1, h, w)
    mask_expanded = torch.nn.functional.interpolate(
      mask_expanded.float(), 
      scale_factor=p, 
      mode='nearest'
    )
        
    orig_img = img[0]
    mean_val = torch.tensor([-2.11, -2.03, -1.80]).view(3, 1, 1) 
    masked_img = orig_img.clone()
    masked_img = torch.where(mask_expanded[0] == 1, mean_val, masked_img)
        
    combined_img = orig_img.clone()
    combined_img = torch.where(mask_expanded[0] == 1, rec_img[0], combined_img)

  plt.figure(figsize=(15, 5))
  plt.subplot(1, 3, 1)
  show_image(orig_img, "Original")
  plt.subplot(1, 3, 2)
  show_image(masked_img, f"Masked ({int(final_mask_ratio*100)}%)")
  plt.subplot(1, 3, 3)
  show_image(combined_img, "Reconstruction")
  plt.tight_layout()
  
  base_name = os.path.basename(actual_path).split('.')[0] if image_path else "cifar"
  plt.savefig(f"reconstruction_{base_name}_{int(final_mask_ratio*100)}.png")
  plt.show()

if __name__ == "__main__":
  path = None
  ratio = None
  
  if len(sys.argv) > 1:
    try:
      ratio = float(sys.argv[1]) / 100.0
    except ValueError:
      path = sys.argv[1]
      
  if len(sys.argv) > 2:
    if path is not None:
      try:
        ratio = float(sys.argv[2]) / 100.0
      except ValueError:
        pass

  visualize(path, ratio)
