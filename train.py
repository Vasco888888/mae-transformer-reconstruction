import torch
from torch.utils.data import DataLoader
from models.mae import MaskedAutoencoderViT
from utils.datasets import build_dataset
from utils.lr_sched import adjust_learning_rate

import yaml

class ConfigNode:
  def __init__(self, data):
    for key, value in data.items():
      if isinstance(value, dict):
        setattr(self, key, ConfigNode(value))
      else:
        setattr(self, key, value)

# Hyperparameters
class Args:
  def __init__(self, config_path="experiments/config.yaml"):
    with open(config_path, "r") as f:
      config = yaml.safe_load(f)
    
    self.model = ConfigNode(config.get("model", {}))
    self.train = ConfigNode(config.get("train", {}))
    self.data = ConfigNode(config.get("data", {}))
    
    # Auto-fallback for device
    if self.train.device == "cuda" and not torch.cuda.is_available():
      self.train.device = "cpu"

args = Args()

def train():
  # Setup Data
  dataset = build_dataset(args)
  data_loader = DataLoader(dataset, batch_size=args.data.batch_size, shuffle=True, num_workers=args.data.num_workers)

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

  # AdamW Optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
    
  # Mixed Precision Scaler
  scaler = torch.cuda.amp.GradScaler()

  print(f"Starting training on {args.train.device}...")

  for epoch in range(args.train.epochs):
    model.train()
    total_loss = 0
        
    for i, (imgs, _) in enumerate(data_loader):
      # Move images to GPU
      imgs = imgs.to(args.train.device)

      # Adjust LR based on the Cosine Schedule
      adjust_learning_rate(optimizer, epoch + i / len(data_loader), args.train)

      # Forward pass with Mixed Precision
      with torch.cuda.amp.autocast():
        pred, mask = model(imgs, mask_ratio=args.train.mask_ratio)
        loss = model.forward_loss(imgs, pred, mask)

      # Backward pass
      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{args.train.epochs}] - Loss: {avg_loss:.4f}")

    # Save the model periodically
    if (epoch + 1) % 10 == 0:
      torch.save(model.state_dict(), f"mae_checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
  train()