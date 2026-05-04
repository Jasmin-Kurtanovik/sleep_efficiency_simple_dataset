"""
Testing and evaluation pipeline for LSTM model without attention.
Generates per-user evaluation metrics and plots.
"""
from pathlib import Path
import json
import os
import sys
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, USER_EMB_DIM, BATCH_SIZE, DATA_PATH, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
from model_without_att import LSTMModelNoAttention
from data_preprocessor import SleepDataPreprocessor, SleepDataset, split_data_by_user
from evaluate_per_user import PerUserEvaluator
from visualize_per_user import PerUserVisualizer


MODEL_SAVE_PATH = "models/best_model_without_att.pt"


def set_seed(seed: int) -> None:
    """Make the train/validation split and optimization reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main evaluation pipeline."""
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
    
    print(f"\nTest sequences: {X_test.shape}")
    
    # Create test dataset and dataloader
    test_dataset = SleepDataset(X_test, y_test, seq_user_idx=user_idx_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create and load model
    input_size = X.shape[2]
    num_users = len(metadata['users'])
    
    model = LSTMModelNoAttention(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        dropout=DROPOUT,
        num_users=num_users,
        user_emb_dim=USER_EMB_DIM
    ).to(device)
    
    print(f"\nLoading model from {MODEL_SAVE_PATH}...")
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_SAVE_PATH}. Run train_validate_without_att.py first.")
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_seq_to_user = [metadata["users"][u_idx] for u_idx in user_idx_test]
    
    evaluator = PerUserEvaluator(model, device=str(device))
    per_user_metrics, aggregated_metrics, all_predictions, all_actuals, user_predictions_dict, user_actuals_dict = \
        evaluator.evaluate_with_user_tracking(test_loader, test_seq_to_user)
    
    # Print metrics
    print("\n" + "="*60)
    print("AGGREGATED METRICS (Model without Attention)")
    print("="*60)
    print(f"RMSE:  {aggregated_metrics['rmse']:.6f}")
    print(f"MAE:   {aggregated_metrics['mae']:.6f}")
    print(f"R²:    {aggregated_metrics['r2']:.6f}")
    
    print("\n" + "="*60)
    print("PER-USER METRICS")
    print("="*60)
    for user_id, metrics in per_user_metrics.items():
        print(f"\nUser {user_id}:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  R²:   {metrics['r2']:.6f}")
        print(f"  N sequences: {metrics['n_sequences']}")
    
    # Save evaluation results
    os.makedirs("evaluation_results", exist_ok=True)
    results_file = "evaluation_results/evaluation_results_without_att.json"
    results = {
        "model": "LSTMModelNoAttention",
        "aggregated_metrics": aggregated_metrics,
        "per_user_metrics": per_user_metrics
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {results_file}")
    
    # Create per-user plots
    print("\nGenerating per-user plots...")
    visualizer = PerUserVisualizer(output_dir="evaluation_results_without_att")
    visualizer.create_individual_user_plots(
        user_predictions_dict,
        user_actuals_dict,
        per_user_metrics,
        model_name="LSTM without Attention"
    )
    print(f"Per-user plots saved to evaluation_results_without_att/plots_per_user/")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
