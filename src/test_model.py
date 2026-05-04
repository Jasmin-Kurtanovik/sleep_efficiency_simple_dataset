"""Test the trained model and create per-user prediction plots."""

from pathlib import Path
import json
import logging
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (  # noqa: E402
    BATCH_SIZE,
    DATA_PATH,
    DROPOUT,
    EVALUATION_RESULTS_DIR,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    MODEL_SAVE_PATH,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    USER_EMB_DIM,
    VAL_RATIO,
    get_device,
)
from data_preprocessor import SleepDataPreprocessor, SleepDataset, split_data_by_user  # noqa: E402
from evaluate_per_user import PerUserEvaluator  # noqa: E402
from model import LSTMModel  # noqa: E402
from visualize_per_user import PerUserVisualizer  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Make the test split reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Run model testing and save per-user plots."""
    set_seed(RANDOM_SEED)

    model_path = Path(MODEL_SAVE_PATH)
    data_path = Path(DATA_PATH)
    output_dir = Path(EVALUATION_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(model_path)

    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        raise FileNotFoundError(data_path)

    device = get_device()

    logger.info("Loading and preprocessing data...")
    preprocessor = SleepDataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    X, y, metadata = preprocessor.load_and_prepare(str(data_path))

    logger.info("Splitting data by user...")
    (_, _, _), (_, _, _), (X_test, y_test, user_idx_test) = split_data_by_user(
        X,
        y,
        metadata,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    test_dataset = SleepDataset(X_test, y_test, seq_user_idx=user_idx_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        num_users=len(metadata["users"]),
        user_emb_dim=USER_EMB_DIM,
    )

    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    logger.info(f"Model loaded from {model_path}")

    test_seq_to_user = [metadata["users"][u_idx] for u_idx in user_idx_test]

    evaluator = PerUserEvaluator(model, device=str(device))
    per_user_metrics, aggregated_metrics, _, _, user_predictions, user_actuals = evaluator.evaluate_with_user_tracking(
        test_loader,
        test_seq_to_user=test_seq_to_user,
    )

    results = {
        "per_user": per_user_metrics,
        "aggregated": aggregated_metrics,
        "config": {
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "lookback_days": SEQUENCE_LENGTH,
            "dropout": DROPOUT,
        },
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info(f"Results saved to {results_path}")

    logger.info("Generating per-user plots...")
    visualizer = PerUserVisualizer(output_dir=str(output_dir))
    visualizer.create_individual_user_plots(user_predictions, user_actuals, per_user_metrics, model_name="LSTM with Attention")

    logger.info(f"Per-user plots saved to {output_dir / 'plots_per_user'}")
    return per_user_metrics, aggregated_metrics, metadata


if __name__ == "__main__":
    main()
