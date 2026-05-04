"""
Training and validation pipeline for LSTM model without attention.
"""
from datetime import datetime
from pathlib import Path
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (  # noqa: E402
    DATA_PATH,
    SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, USER_EMB_DIM,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)
from model_without_att import LSTMModelNoAttention
from data_preprocessor import SleepDataPreprocessor, SleepDataset, split_data_by_user


def set_seed(seed: int) -> None:
    """Make the train/validation split and optimization reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Training and validation loop for LSTM model without attention."""
    
    def __init__(self, model, device, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, y, user_idx = batch
            x = x.to(self.device)
            y = y.to(self.device).view(-1, 1)
            user_idx = user_idx.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x, user_idx)
            loss = self.criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y, user_idx = batch
                x = x.to(self.device)
                y = y.to(self.device).view(-1, 1)
                user_idx = user_idx.to(self.device)
                
                outputs = self.model(x, user_idx)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
        
        return total_loss / len(val_loader.dataset)
    
    def fit(self, train_loader, val_loader, max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
        """Fit the model with early stopping."""
        patience_counter = 0
        
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  [NEW BEST] New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model


def main():
    """Main training pipeline."""
    set_seed(RANDOM_SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = SleepDataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    X, y, metadata = preprocessor.load_and_prepare(DATA_PATH)
    print(f"Data shape: {X.shape}")
    print(f"Number of unique users: {len(metadata['users'])}")
    
    # Split data by user
    (X_train, y_train, user_idx_train), (X_val, y_val, user_idx_val), (X_test, y_test, user_idx_test) = \
        split_data_by_user(X, y, metadata, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
    
    print(f"\nTrain sequences: {X_train.shape}")
    print(f"Val sequences: {X_val.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = SleepDataset(X_train, y_train, seq_user_idx=user_idx_train)
    val_dataset = SleepDataset(X_val, y_val, seq_user_idx=user_idx_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_size = X.shape[2]  # Number of features
    num_users = len(metadata['users'])  # Total number of unique users
    
    model = LSTMModelNoAttention(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        dropout=DROPOUT,
        num_users=num_users,
        user_emb_dim=USER_EMB_DIM
    ).to(device)
    
    print(f"\nModel architecture (no attention):")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  User embedding dim: {USER_EMB_DIM}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\nStarting training...")
    trainer = Trainer(model, device, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model = trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING_PATIENCE)
    
    # Save model and training history
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model_without_att.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Verify checkpoint was saved
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not saved at {model_path}")
    
    # Save training history metadata
    history_file = f"models/history_without_att_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    history = {
        "best_val_loss": trainer.best_val_loss,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "user_emb_dim": USER_EMB_DIM,
        "lookback_days": SEQUENCE_LENGTH,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE
    }
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_file}")


if __name__ == "__main__":
    main()
