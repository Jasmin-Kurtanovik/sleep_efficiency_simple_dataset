"""Train and validate the LSTM model."""

from datetime import datetime
from pathlib import Path
import json
import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (  # noqa: E402
    BATCH_SIZE,
    DATA_PATH,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_SIZE,
    HISTORY_SAVE_DIR,
    INPUT_SIZE,
    LEARNING_RATE,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    MODEL_SAVE_PATH,
    NUM_LAYERS,
    OUTPUT_SIZE,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    USER_EMB_DIM,
    VAL_RATIO,
    WEIGHT_DECAY,
    get_device,
    print_config,
)
from data_preprocessor import SleepDataPreprocessor, SleepDataset, split_data_by_user  # noqa: E402
from model import LSTMModel  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Make the train/validation split and optimization reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Trainer class for the LSTM model."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = None,
        weight_decay: float = None,
    ):
        if learning_rate is None:
            learning_rate = LEARNING_RATE
        if weight_decay is None:
            weight_decay = WEIGHT_DECAY

        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=LR_SCHEDULER_PATIENCE,
        )
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            if len(batch) == 3:
                x, y, user_idx = batch
                user_idx = user_idx.to(self.device)
            else:
                x, y = batch
                user_idx = None

            x = x.to(self.device)
            y = y.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()
            predictions = self.model(x, user_idx)
            loss = self.criterion(predictions, y)
            loss.backward()
            self.optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size

        # Return average loss per sample
        return total_loss / max(len(train_loader.dataset), 1)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate on the validation split."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, user_idx = batch
                    user_idx = user_idx.to(self.device)
                else:
                    x, y = batch
                    user_idx = None

                x = x.to(self.device)
                y = y.to(self.device).view(-1, 1)

                predictions = self.model(x, user_idx)
                loss = self.criterion(predictions, y)

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size

        return total_loss / max(len(val_loader.dataset), 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_path: str = MODEL_SAVE_PATH,
        patience: int = EARLY_STOPPING_PATIENCE,
    ):
        """Train the model and keep the best validation checkpoint."""
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f} (Best)"
                )
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history


def main():
    """Run the training and validation pipeline."""
    set_seed(RANDOM_SEED)
    print_config()

    device = get_device()

    logger.info("Loading sleep diary data...")
    preprocessor = SleepDataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    X, y, metadata = preprocessor.load_and_prepare(DATA_PATH)

    (X_train, y_train, user_idx_train), (X_val, y_val, user_idx_val), _ = split_data_by_user(
        X,
        y,
        metadata,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    train_dataset = SleepDataset(X_train, y_train, seq_user_idx=user_idx_train)
    val_dataset = SleepDataset(X_val, y_val, seq_user_idx=user_idx_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        num_users=len(metadata["users"]),
        user_emb_dim=USER_EMB_DIM,
    )

    trainer = Trainer(model, device=str(device))
    history = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS, model_path=MODEL_SAVE_PATH)

    # Ensure a checkpoint exists even if the best-model save path was interrupted.
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), MODEL_SAVE_PATH)
    if not Path(MODEL_SAVE_PATH).exists():
        raise FileNotFoundError(f"Failed to save model checkpoint to {MODEL_SAVE_PATH}")

    Path(HISTORY_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    history_path = Path(HISTORY_SAVE_DIR) / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    logger.info(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    logger.info(f"History saved to {history_path}")

    return history


if __name__ == "__main__":
    main()
