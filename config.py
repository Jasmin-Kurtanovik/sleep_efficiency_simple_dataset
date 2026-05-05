"""Project-wide configuration for LSTM training and evaluation."""

from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = str(ROOT_DIR / "data" / "sleep_diary.csv")
MODEL_SAVE_PATH = str(ROOT_DIR / "models" / "best_model.pt")
HISTORY_SAVE_DIR = str(ROOT_DIR / "models" / "history")
EVALUATION_RESULTS_DIR = str(ROOT_DIR / "evaluation_results")
PLOTS_SAVE_DIR = str(ROOT_DIR / "evaluation_results" / "plots_per_user")

MODEL_SAVE_PATH_TRANSFORMER = str(ROOT_DIR / "models" / "best_model_transformer.pt")
HISTORY_SAVE_DIR_TRANSFORMER = str(ROOT_DIR / "models" / "history_transformer")
EVALUATION_RESULTS_DIR_TRANSFORMER = str(ROOT_DIR / "evaluation_results_transformer")
PLOTS_SAVE_DIR_TRANSFORMER = str(ROOT_DIR / "evaluation_results_transformer" / "plots_per_user")

# Most important experiment knobs.
LOOKBACK_DAYS = 7
SEQUENCE_LENGTH = 7
INPUT_SIZE = 6
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 4
DROPOUT = 0.2
USER_EMB_DIM = 16

TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FF_DIM = 128
TRANSFORMER_MAX_SEQ_LEN = 32

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 3

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
RANDOM_SEED = 42


def get_device() -> torch.device:
    """Return the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config_dict() -> dict:
    """Return the main configuration values as a plain dictionary."""
    return {
        "data_path": DATA_PATH,
        "model_save_path": MODEL_SAVE_PATH,
        "history_save_dir": HISTORY_SAVE_DIR,
        "evaluation_results_dir": EVALUATION_RESULTS_DIR,
        "plots_save_dir": PLOTS_SAVE_DIR,
        "model_save_path_transformer": MODEL_SAVE_PATH_TRANSFORMER,
        "history_save_dir_transformer": HISTORY_SAVE_DIR_TRANSFORMER,
        "evaluation_results_dir_transformer": EVALUATION_RESULTS_DIR_TRANSFORMER,
        "plots_save_dir_transformer": PLOTS_SAVE_DIR_TRANSFORMER,
        "lookback_days": LOOKBACK_DAYS,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "user_emb_dim": USER_EMB_DIM,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
        "transformer_num_heads": TRANSFORMER_NUM_HEADS,
        "transformer_num_layers": TRANSFORMER_NUM_LAYERS,
        "transformer_ff_dim": TRANSFORMER_FF_DIM,
        "transformer_max_seq_len": TRANSFORMER_MAX_SEQ_LEN,
        "device": str(get_device()),
    }


def print_config() -> None:
    """Print the current experiment configuration."""
    config = get_config_dict()
    print("\nCurrent configuration:\n" + "-" * 32)
    for key in [
        "lookback_days",
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "user_emb_dim",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "early_stopping_patience",
        "train_ratio",
        "val_ratio",
        "random_seed",
        "transformer_num_heads",
        "transformer_num_layers",
        "transformer_ff_dim",
        "device",
    ]:
        print(f"{key}: {config[key]}")
