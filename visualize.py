import torch
import matplotlib.pyplot as plt
from models.mae import MaskedAutoencoderViT, unpatchify
from utils.datasets import build_dataset
from train import Args

def show_image(image, title=''):
  """
  Image is [3, 224, 224] tensor, typically normalized.
  Needs un-normalizing back to [0, 1].
  """
  # Inverse normalize
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  image = image.cpu() * std + mean
  image = torch.clamp(image, 0, 1)
    
  # Plot
  plt.imshow(image.permute(1, 2, 0).numpy())
  plt.title(title, fontsize=16)
  plt.axis('off')

def visualize():
  args = Args()
  args.train.device = "cpu"  # Usually enough for inference visualization
    
  # Setup Data
  dataset = build_dataset(args)
    
  # Pick an image from dataset (change index to see others)
  img, _ = dataset[0]  
  img = img.unsqueeze(0).to(args.train.device)  # Add batch dimension

  # Setup Model
  model = MaskedAutoencoderViT(
    patch_size=args.model.patch_size, 
    embed_dim=args.model.embed_dim, 
    depth=args.model.depth, 
    num_heads=args.model.num_heads,
    decoder_embed_dim=args.model.decoder_embed_dim, 
    decoder_depth=args.model.decoder_depth, 
    decoder_num_heads=args.model.decoder_num_heads,
  ).to(args.train.device)

  # Note: If you have a trained model, uncomment the following line!
  # model.load_state_dict(torch.load("mae_checkpoint_epoch_50.pth", map_location="cpu"))
  model.eval()

  # Get predictions
  with torch.no_grad():
    # Forward pass (pred contains the reconstructed patches)
    pred, mask = model(img, mask_ratio=args.train.mask_ratio)
        
    # Unpatchify the predictions to an image shape: [B, 3, H, W]
    rec_img = model.unpatchify(pred)
        
    # Create masked image
    # Mask is 1 for dropped patches, 0 for kept
    p = model.patch_size
    h = w = img.shape[2] // p
        
    # Convert [B, N] mask to [B, 1, H, w]
    mask_expanded = mask.reshape(img.shape[0], 1, h, w)
    # Scale nearest neighbor to [B, 1, H * p, W * p]
    mask_expanded = torch.nn.functional.interpolate(
      mask_expanded.float(), 
      scale_factor=p, 
      mode='nearest'
    )
        
    # 1. Original image
    orig_img = img[0]
        
    # 2. Masked image: gray out the dropped patches
    # Normalized gray value
    mean_val = torch.tensor([-2.11, -2.03, -1.80]).view(3, 1, 1) 
    masked_img = orig_img.clone()
    masked_img = torch.where(mask_expanded[0] == 1, mean_val, masked_img)
        
    # 3. Reconstructed image: Combine original and predicted
    # Use original image for kept patches, and pred for dropped patches
    combined_img = orig_img.clone()
    combined_img = torch.where(mask_expanded[0] == 1, rec_img[0], combined_img)

    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    show_image(orig_img, "Original")
    
    plt.subplot(1, 3, 2)
    show_image(masked_img, f"Masked ({int(args.train.mask_ratio*100)}%)")
    
    plt.subplot(1, 3, 3)
    show_image(combined_img, "Reconstruction")
    
    plt.tight_layout()
    plt.savefig("reconstruction_demo.png")
    print("Saved visualization to reconstruction_demo.png")
    
    # Also show it if in a notebook
    plt.show()

if __name__ == "__main__":
  visualize()
