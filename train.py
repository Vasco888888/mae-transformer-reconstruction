import torch
from torch.utils.data import DataLoader
from models.mae import MaskedAutoencoderViT
from utils.datasets import build_dataset
from utils.lr_sched import adjust_learning_rate

# Hyperparameters
class Args:
  batch_size = 64
  epochs = 50
  warmup_epochs = 5
  lr = 1.5e-4
  min_lr = 0.
  weight_decay = 0.05
  mask_ratio = 0.75
  device = "cuda" if torch.cuda.is_available() else "cpu"

args = Args()

def train():
  # Setup Data
  dataset = build_dataset(args)
  data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

  # Setup Model
  model = MaskedAutoencoderViT(
    patch_size=16, embed_dim=768, depth=12, num_heads=12,
    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
  ).to(args.device)

  # AdamW Optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
  # Mixed Precision Scaler
  scaler = torch.cuda.amp.GradScaler()

  print(f"Starting training on {args.device}...")

  for epoch in range(args.epochs):
    model.train()
    total_loss = 0
        
    for i, (imgs, _) in enumerate(data_loader):
      # Move images to GPU
      imgs = imgs.to(args.device)

      # Adjust LR based on the Cosine Schedule
      adjust_learning_rate(optimizer, epoch + i / len(data_loader), args)

      # Forward pass with Mixed Precision
      with torch.cuda.amp.autocast():
        pred, mask = model(imgs, mask_ratio=args.mask_ratio)
        loss = model.forward_loss(imgs, pred, mask)

      # Backward pass
      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

    # Save the model periodically
    if (epoch + 1) % 10 == 0:
      torch.save(model.state_dict(), f"mae_checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
  train()