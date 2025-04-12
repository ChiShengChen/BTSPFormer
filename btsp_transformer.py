import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import argparse
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# -----------------------------
# CSV Logger callback
# -----------------------------
class CSVLogger(pl.Callback):
    def __init__(self, save_dir='csv_logs/btspformer_new'):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.filename = os.path.join(
            save_dir,
            f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'train_mse',
                'train_mae',
                'val_loss',
                'val_mse',
                'val_mae',
                'learning_rate',
                'best_val_loss',
                'best_val_mse',
                'best_val_mae'
            ])

        self.metrics = []
        self.best_val_loss = float('inf')
        self.best_val_mse = float('inf')
        self.best_val_mae = float('inf')

    def on_train_epoch_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get('val_loss')
        current_val_mse = trainer.callback_metrics.get('val_mse')
        current_val_mae = trainer.callback_metrics.get('val_mae')

        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        if current_val_mse is not None and current_val_mse < self.best_val_mse:
            self.best_val_mse = current_val_mse
        if current_val_mae is not None and current_val_mae < self.best_val_mae:
            self.best_val_mae = current_val_mae

        metrics = {
            'epoch': trainer.current_epoch + 1,
            'train_loss': trainer.callback_metrics.get('train_loss_epoch'), # Use epoch metric
            'train_mse': trainer.callback_metrics.get('train_mse_epoch'),  # Use epoch metric
            'train_mae': trainer.callback_metrics.get('train_mae_epoch'),  # Use epoch metric
            'val_loss': current_val_loss,
            'val_mse': current_val_mse,
            'val_mae': current_val_mae,
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'],
            'best_val_loss': self.best_val_loss,
            'best_val_mse': self.best_val_mse,
            'best_val_mae': self.best_val_mae
        }

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k] for k in [
                'epoch',
                'train_loss',
                'train_mse',
                'train_mae',
                'val_loss',
                'val_mse',
                'val_mae',
                'learning_rate',
                'best_val_loss',
                'best_val_mse',
                'best_val_mae'
            ] if metrics[k] is not None]) # Handle None values gracefully


# -----------------------------
# BTSP-Inspired Attention Layer
# -----------------------------
class BTSPAttention(nn.Module):
    def __init__(self, model_dim, heads):
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads

        assert self.model_dim % self.heads == 0, "model_dim must be divisible by heads"

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        # Learnable parameters for time decay and importance sampling
        self.et_gate = nn.Parameter(torch.randn(1, 1, 1))  # Exponential Time decay gate
        self.is_gate = nn.Parameter(torch.randn(1, 1, 1))  # Importance Sampling gate (bias)
        # Using a fixed but potentially long range bias, learnable scaling/shifting happens via gates
        self.time_bias = nn.Parameter(torch.linspace(-1, 1, steps=500).unsqueeze(0).unsqueeze(0), requires_grad=False) # Increased steps

    def forward(self, x):
        B, T, D = x.shape
        Q = self.query(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2) # B, H, T, Dh
        K = self.key(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2) # B, H, T, Dh
        V = self.value(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2) # B, H, T, Dh

        # Standard scaled dot-product attention score
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) # B, H, T, T

        # Create relative position matrix (indices difference)
        position = torch.arange(T, device=x.device).unsqueeze(0) - torch.arange(T, device=x.device).unsqueeze(1) # T, T
        # Map relative positions to the time_bias range (-1 to 1)
        # Ensure indices stay within bounds of time_bias parameter
        indices = torch.clamp((position + self.time_bias.shape[-1]//2), 0, self.time_bias.shape[-1]-1).long() # T, T
        # Select biases based on relative positions
        time_penalty = self.time_bias[:, :, indices].squeeze(1) # T, T
        time_penalty = time_penalty.unsqueeze(0).unsqueeze(1) # 1, 1, T, T broadcastable to B, H, T, T

        # Apply learnable gates to time penalty and add bias
        scores = scores + torch.sigmoid(self.et_gate) * time_penalty + self.is_gate # B, H, T, T

        attn_weights = F.softmax(scores, dim=-1) # B, H, T, T
        context = attn_weights @ V # B, H, T, Dh
        context = context.transpose(1, 2).reshape(B, T, D) # B, T, H*Dh = D

        return self.out_proj(context)

# -----------------------------
# Transformer Block with BTSPAttention
# -----------------------------
class BTSPFormerBlock(nn.Module):
    def __init__(self, model_dim, heads, dropout=0.1):
        super().__init__()
        self.attn = BTSPAttention(model_dim, heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# -----------------------------
# BTSPFormer Model
# -----------------------------
class BTSPFormer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, heads=4, layers=2, output_dim=1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList([
            BTSPFormerBlock(model_dim, heads) for _ in range(layers)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, output_dim)
        )

    def forward(self, x):
        x = self.input_embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x[:, -1, :])

# -----------------------------
# Lightning Module
# -----------------------------
class LitBTSPFormer(pl.LightningModule):
    def __init__(self, input_dim=1, model_dim=64, heads=4, layers=2, output_dim=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters() # Saves init args to hparams
        self.model = BTSPFormer(input_dim, model_dim, heads, layers, output_dim)
        self.lr = lr
        self.loss_fn = nn.MSELoss()

        self.plot_epochs = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100} # Epochs to plot predictions
        self.best_val_loss = float('inf')
        self.plot_data = None # To store a batch for plotting

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # Grab a batch from validation loader for plotting
        if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders:
             val_loader = self.trainer.val_dataloaders
             try: # Iterate safely
                 for batch in val_loader:
                     self.plot_data = (batch[0].cpu().numpy(), batch[1].cpu().numpy())
                     break
             except Exception as e:
                 print(f"Could not get plot data from val_loader: {e}")
                 self.plot_data = None # Ensure it's None if loading fails


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.squeeze()
        loss = self.loss_fn(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.squeeze()
        loss = self.loss_fn(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mse", mse, on_epoch=True)
        self.log("val_mae", mae, on_epoch=True)

        if loss < self.best_val_loss:
             self.best_val_loss = loss

        return loss

    def plot_predictions(self, x_np, y_true_np, y_pred_np, epoch):
        plt.figure(figsize=(10, 6))

        # Extract the last time step's feature from input x as the time axis
        x_time = x_np[:, -1, 0].reshape(-1)
        y_true = y_true_np.reshape(-1)
        y_pred = y_pred_np.reshape(-1)

        # Sort points based on the time axis for a cleaner plot
        sort_idx = np.argsort(x_time)
        x_time_sorted = x_time[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        plt.plot(x_time_sorted, y_true_sorted, 'b-', label='True', alpha=0.6)
        plt.plot(x_time_sorted, y_pred_sorted, 'r--', label='Predicted', alpha=0.6)
        plt.title(f'BTSPFormer Predictions vs True - Epoch {epoch}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Amplitude (normalized)')
        plt.legend()
        plt.grid(True)

        plot_dir = 'btspformer_new_plots'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'epoch_{epoch}.png'))
        plt.close()

    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        if epoch in self.plot_epochs and self.plot_data is not None:
            x_np, y_true_np = self.plot_data
            x_tensor = torch.FloatTensor(x_np).to(self.device)
            with torch.no_grad():
                self.model.eval() # Set model to eval mode for prediction
                y_pred = self(x_tensor).squeeze().cpu().numpy()
                self.model.train() # Set back to train mode

            self.plot_predictions(x_np, y_true_np.squeeze(), y_pred, epoch)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01 # Added weight decay like in reference
        )
        # Added scheduler like in reference
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Monitor validation loss
            },
        }

# -----------------------------
# Data generation
# -----------------------------
def generate_damped_sine(sequence_length=50, num_samples=5000):
    A, gamma, omega, phi = 1.0, 0.1, 2.0, 0
    t = np.linspace(0, 20, num_samples)
    x = A * np.exp(-gamma * t) * np.cos(omega * t + phi)
    t = (t - t.mean()) / t.std()
    x = (x - x.mean()) / x.std()

    stride = 1
    n_sequences = (len(t) - sequence_length) // stride
    indices = np.arange(n_sequences) * stride

    X = np.array([t[i:i+sequence_length] for i in indices])
    Y = x[indices + sequence_length]

    X_tensor = torch.FloatTensor(X).reshape(-1, sequence_length, 1)
    Y_tensor = torch.FloatTensor(Y).reshape(-1, 1)

    return TensorDataset(X_tensor, Y_tensor)

# -----------------------------
# Training script
# -----------------------------
def train_model(args): # Changed to accept args
    pl.seed_everything(42) # Added seed for reproducibility

    dataset = generate_damped_sine(args.seq_len, args.num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers, # Use args
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers, # Use args
        pin_memory=True
    )

    # Initialize the Lightning Module directly
    lit_model = LitBTSPFormer(
        input_dim=1,
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        output_dim=1,
        lr=args.lr
    )

    # Callbacks setup
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints/btspformer_new', # Specific directory
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        CSVLogger(save_dir='csv_logs/btspformer_new') # Use the new logger
    ]

    # TensorBoard Logger setup
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
        name='btspformer_new', # Specific name
        version=None,
        default_hp_metric=False
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() and args.use_gpu else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() and args.use_gpu else 32,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=tb_logger, # Use TensorBoard logger
        log_every_n_steps=20,
        gradient_clip_val=1.0, # Added gradient clipping
        deterministic=False, # Set to False as in reference
    )

    # No need for the 'use_btsp' flag anymore, as we only have one model path now
    print(f"Training BTSPFormer (New)...")
    trainer.fit(lit_model, train_loader, val_loader)

    print(f"Training completed. Best validation loss: {lit_model.best_val_loss:.4f}")

    # Save final model state dict
    final_model_path = "btspformer_new_model.pth"
    torch.save(lit_model.state_dict(), final_model_path)
    print(f"Model state dict saved to {final_model_path}")


# Main execution block with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BTSPFormer (New)")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for input")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of samples to generate") # Reduced default for faster testing
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs") # Reduced default
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model_dim", type=int, default=64, help="Model dimension (embedding size)")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()
    train_model(args)

# Remove old training calls
# # Uncomment to train with BTSPFormer
# train_model(use_btsp=True)
#
# # Uncomment to compare with vanilla Transformer
# # train_model(use_btsp=False)
