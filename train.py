import torch
from torch.utils.data import DataLoader
from models.mae import MaskedAutoencoderViT
from utils.datasets import build_dataset
from utils.lr_sched import adjust_learning_rate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import os

class ConfigNode:
  def __init__(self, data):
    for key, value in data.items():
      if isinstance(value, dict):
        setattr(self, key, ConfigNode(value))
      else:
        setattr(self, key, value)

class Args:
  def __init__(self, config_path="experiments/config.yaml"):
    with open(config_path, "r") as f:
      config = yaml.safe_load(f)
    
    self.model = ConfigNode(config.get("model", {}))
    self.train = ConfigNode(config.get("train", {}))
    self.data = ConfigNode(config.get("data", {}))
    
    if self.train.device == "cuda" and not torch.cuda.is_available():
      self.train.device = "cpu"

args = Args()

def train():
  dataset = build_dataset(args)
  data_loader = DataLoader(dataset, batch_size=args.data.batch_size, shuffle=True, num_workers=args.data.num_workers)

  model = MaskedAutoencoderViT(
    patch_size=args.model.patch_size, 
    embed_dim=args.model.embed_dim, 
    depth=args.model.depth, 
    num_heads=args.model.num_heads,
    decoder_embed_dim=args.model.decoder_embed_dim, 
    decoder_depth=args.model.decoder_depth, 
    decoder_num_heads=args.model.decoder_num_heads,
  ).to(args.train.device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
  scaler = torch.cuda.amp.GradScaler()
  writer = SummaryWriter(log_dir=args.train.log_dir)

  print(f"Starting training on {args.train.device}...")

  for epoch in range(args.train.epochs):
    model.train()
    total_loss = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch [{epoch+1}/{args.train.epochs}]")
    for i, (imgs, _) in pbar:
      imgs = imgs.to(args.train.device)
      lr = adjust_learning_rate(optimizer, epoch + i / len(data_loader), args.train)

      with torch.cuda.amp.autocast():
        pred, mask = model(imgs, mask_ratio=args.train.mask_ratio)
        loss = model.forward_loss(imgs, pred, mask)

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item()
      writer.add_scalar('Train/Loss_Step', loss.item(), epoch * len(data_loader) + i)
      writer.add_scalar('Train/LR', lr, epoch * len(data_loader) + i)
      
      if torch.cuda.is_available():
        writer.add_scalar('GPU/Memory_Allocated_MB', torch.cuda.memory_allocated() / 1024**2, epoch * len(data_loader) + i)
        writer.add_scalar('GPU/Memory_Reserved_MB', torch.cuda.memory_reserved() / 1024**2, epoch * len(data_loader) + i)
      
      pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.6e}'})

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{args.train.epochs}] - Average Loss: {avg_loss:.4f}")
    writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)

    if (epoch + 1) % 10 == 0:
      os.makedirs("checkpoints", exist_ok=True)
      torch.save(model.state_dict(), f"checkpoints/mae_checkpoint_epoch_{epoch+1}.pth")
      
      model.eval()
      with torch.no_grad():
        val_imgs, _ = next(iter(data_loader))
        val_imgs = val_imgs[:8].to(args.train.device)
        
        pred, mask = model(val_imgs, mask_ratio=args.train.mask_ratio)
        recon_imgs = model.unpatchify(pred)
        
        m = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
        m = model.unpatchify(m)
        masked_imgs = val_imgs * (1 - m)
        
        from torchvision.utils import make_grid
        img_grid = torch.cat([val_imgs, masked_imgs, recon_imgs], dim=0)
        img_grid = make_grid(img_grid, nrow=8, normalize=True)
        writer.add_image('Training/Reconstructions', img_grid, epoch)
        
      model.train()

  writer.close()

if __name__ == "__main__":
  train()