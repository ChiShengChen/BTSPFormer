import os
os.environ['RDMAV_FORK_SAFE'] = '1'

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
import csv
from datetime import datetime
import argparse

# CSV Logger callback
class CSVLogger(pl.Callback):
    def __init__(self, save_dir='csv_logs/btspformer'):
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
            'train_loss': trainer.callback_metrics.get('train_loss'),
            'train_mse': trainer.callback_metrics.get('train_mse'),
            'train_mae': trainer.callback_metrics.get('train_mae'),
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
            ]])

# Generate damped sine wave data function
def generate_damped_sine(sequence_length=50, num_samples=5000):
    A, gamma, omega, phi = 1.0, 0.1, 2.0, 0
    t = np.linspace(0, 20, num_samples)
    x = A * np.exp(-gamma * t) * np.cos(omega * t + phi)
    
    # Normalize data
    t = (t - t.mean()) / t.std()
    x = (x - x.mean()) / x.std()
    
    stride = 1
    n_sequences = (len(t) - sequence_length) // stride
    indices = np.arange(n_sequences) * stride
    
    X = np.array([t[i:i+sequence_length] for i in indices])
    Y = x[indices + sequence_length]
    
    X_tensor = torch.FloatTensor(X).reshape(-1, sequence_length, 1)
    Y_tensor = torch.FloatTensor(Y).reshape(-1, 1)
    
    return torch.utils.data.TensorDataset(X_tensor, Y_tensor)

# BTSP Transformer Model
class BTSPFormer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, heads=4, layers=2, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        x = self.input_embedding(x)  # [batch_size, sequence_length, model_dim]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take the last token
        return self.output_layer(x)

# Lightning Module
class LitBTSPFormer(pl.LightningModule):
    def __init__(self, input_dim=1, model_dim=64, heads=4, layers=2, output_dim=1, lr=1e-3):
        super().__init__()
        self.model = BTSPFormer(input_dim, model_dim, heads, layers, output_dim)
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()
        
        self.plot_epochs = {1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100}
        self.best_val_loss = float('inf')
        self.plot_data = None
    
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        val_loader = self.trainer.val_dataloaders
        for batch in val_loader:
            self.plot_data = (batch[0].cpu().numpy(), batch[1].cpu().numpy())
            break
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        
        mse = F.mse_loss(y_hat, y.squeeze())
        mae = F.l1_loss(y_hat, y.squeeze())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        
        mse = F.mse_loss(y_hat, y.squeeze())
        mae = F.l1_loss(y_hat, y.squeeze())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        
        return loss

    def plot_predictions(self, x, y_true, y_pred, epoch):
        plt.figure(figsize=(10, 6))
        
        x = x[:, -1, 0].reshape(-1)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y_true = y_true[sort_idx]
        y_pred = y_pred[sort_idx]
        
        plt.plot(x, y_true, 'b-', label='True', alpha=0.5)
        plt.plot(x, y_pred, 'r--', label='Predicted', alpha=0.5)
        plt.title(f'Damped Sine - Epoch {epoch}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Amplitude (normalized)')
        plt.legend()
        plt.grid(True)
        
        os.makedirs('btspformer_plots', exist_ok=True)
        plt.savefig(f'btspformer_plots/epoch_{epoch}.png')
        plt.close()

    def compare_sorted_unsorted_predictions(self, x, y_true, y_pred, epoch):
        x_last = x[:, -1, 0].reshape(-1)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_last, y_true, 'b-', label='True')
        plt.plot(x_last, y_pred, 'r--', label='Predicted')
        plt.title(f'[Unsorted] Epoch {epoch}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sort_idx = np.argsort(x_last)
        x_sorted = x_last[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        plt.plot(x_sorted, y_true_sorted, 'b-', label='True (sorted)')
        plt.plot(x_sorted, y_pred_sorted, 'r--', label='Predicted (sorted)')
        plt.title(f'[Sorted] Epoch {epoch}')
        plt.xlabel('Time (sorted)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        os.makedirs('btspformer_plots/compare', exist_ok=True)
        plt.savefig(f'btspformer_plots/compare/epoch_{epoch}_compare.png')
        plt.close()

    def on_train_epoch_end(self):
        if self.current_epoch + 1 in self.plot_epochs and self.plot_data is not None:
            x, y_true = self.plot_data
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                y_pred = self(x_tensor).squeeze().cpu().numpy()
            
            self.plot_predictions(x, y_true, y_pred, self.current_epoch + 1)
            self.compare_sorted_unsorted_predictions(x_tensor.cpu().numpy(), y_true, y_pred, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
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
                "monitor": "val_loss",
            },
        }

def main(args):
    pl.seed_everything(42)
    
    # Generate data
    dataset = generate_damped_sine(args.seq_len, args.num_samples)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = LitBTSPFormer(
        input_dim=1, 
        model_dim=args.model_dim, 
        heads=args.heads, 
        layers=args.layers, 
        output_dim=1,
        lr=args.lr
    )
    
    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/btspformer',
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min'
        ),
        CSVLogger(save_dir='csv_logs/btspformer')
    ]
    
    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() and args.use_gpu else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() and args.use_gpu else 32,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir='logs/',
                name='btspformer',
                version=None,
                default_hp_metric=False
            )
        ],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        deterministic=False,
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Best validation loss: {model.best_val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "btspformer_model.pth")
    print("Model saved to btspformer_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    main(args)
