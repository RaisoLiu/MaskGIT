import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS, output_path):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training(output_path)
        self.writer_init()
        self.vocab_size = self.model.transformer.vocab_size # 1025 = 1024 + 1 (mask token)

    def writer_init(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter('logs')

    def write_loss(self, loss, epoch, mode='train'):
        self.writer.add_scalar(f'Loss/{mode}', loss, epoch)

    def close_writer(self):
        self.writer.close()
        
    @staticmethod
    def prepare_training(output_path):
        os.makedirs(output_path, exist_ok=True)

    def train_one_epoch(self, train_loader, epoch, args):
        losses = []
        self.model.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for i, x in enumerate(pbar):
            x = x.to(args.device)
            logits, ground_truth = self.model(x) # logits: [batch_size, seq_len, vocab_size], ground_truth: [batch_size, seq_len]
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), ground_truth.view(-1))
            loss.backward()
            losses.append(loss.detach().cpu().item())

            if i % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

            pbar.set_postfix({'loss': f'{np.mean(losses):.4f}'})
            
        self.writer.add_scalar("loss/train", np.mean(losses), epoch)
        return np.mean(losses)

    def eval_one_epoch(self, val_loader, epoch, args):
        losses = []
        self.model.eval()

        for x in val_loader:
            x = x.to(args.device)
            logits, ground_truth = self.model(x)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), ground_truth.view(-1))
            losses.append(loss.detach().cpu().item())

        print(f"epoch: {epoch} / {args.epochs}, loss: {np.mean(losses)}")
        self.writer.add_scalar("loss/val", np.mean(losses), epoch)
        return np.mean(losses)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epochs)
        ])
        return optimizer,scheduler
    

def save_losses(train_losses, val_losses, output_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 繪製損失圖表
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss') 
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 儲存為 PNG
    plt.savefig(f"{output_path}/loss_plot.png", dpi=300, bbox_inches='tight')
    
    # 儲存為 EPS
    plt.savefig(f"{output_path}/loss_plot.eps", format='eps', bbox_inches='tight')
    
    plt.close()
    
    # 儲存為 CSV
    df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    df.to_csv(f"{output_path}/losses.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--out_d_path', type=str, default='./checkpoints/', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=1, help='Number for gradient accumulation at batch level.') 
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of warmup epochs.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    # parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()
    
    from datetime import datetime
    now = datetime.now()
    folder_name = now.strftime("%Y%m%d_%H%M%S")
    
    # Create new folder under out_d_path
    output_path = os.path.join(args.out_d_path, folder_name)
    os.makedirs(output_path, exist_ok=True)

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS, output_path)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
   

    best_val = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f'Training epoch {epoch}/{args.epochs}')
        train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_transformer.writer.add_scalar('Loss/train', train_loss, epoch)
        train_transformer.writer.add_scalar('Loss/val', val_loss, epoch)

        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # Plot loss curves
        save_losses(train_losses, val_losses, output_path)

        # Save model every save_per_epoch epochs
        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), f"{output_path}/epoch_{epoch}.pth")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"{output_path}/best_val.pth")

        if epoch == args.epochs:
            
            # Save last model
            torch.save(train_transformer.model.transformer.state_dict(), f"{output_path}/last.pth")
            # Close writer
            train_transformer.close_writer()

            # Save training configuration
            import yaml
            config_dict = vars(args)
            with open(f"{output_path}/train_args.yml", 'w') as f:
                yaml.dump(config_dict, f)
            


