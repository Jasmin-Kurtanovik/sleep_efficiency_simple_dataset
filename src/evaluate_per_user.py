"""Per-user evaluation helpers for the LSTM model."""
from pathlib import Path
import sys
import json
import logging
from collections import defaultdict
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import LSTMModel
from data_preprocessor import SleepDataPreprocessor, SleepDataset, split_data_by_user
from config import (
    BATCH_SIZE,
    DATA_PATH,
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    USER_EMB_DIM,
    VAL_RATIO,
    get_device,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerUserEvaluator:
    """
    Evaluator that tracks performance on a per-user basis.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
    
    def evaluate_with_user_tracking(
        self,
        test_loader: DataLoader,
        test_seq_to_user: List[int] = None,
        user_id_lookup: List = None,
    ) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, Dict, Dict]:
        """
        Evaluate model with per-user tracking.
        
        Args:
            test_loader: Test data loader
            test_seq_to_user: List mapping test sequence index to user ID
            
        Returns:
            Tuple of (per_user_metrics, aggregated_metrics, all_predictions, all_actuals, user_predictions_dict, user_actuals_dict)
        """
        self.model.eval()
        
        # Track predictions per user
        user_predictions = defaultdict(list)
        user_actuals = defaultdict(list)
        
        all_predictions = []
        all_actuals = []
        sequence_idx = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    x, y, user_idx = batch
                    user_idx = user_idx.to(self.device)
                else:
                    x, y = batch
                    user_idx = None

                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x, user_idx)

                predictions = pred.detach().cpu().view(-1).tolist()
                actuals = y.detach().cpu().view(-1).tolist()

                # Assign predictions to users
                for i, (pred_val, actual_val) in enumerate(zip(predictions, actuals)):
                    all_predictions.append(pred_val)
                    all_actuals.append(actual_val)

                    # Determine user ID: prefer test_seq_to_user if available, else per-batch user_idx
                    user_id = None
                    
                    if test_seq_to_user is not None and sequence_idx < len(test_seq_to_user):
                        user_id = test_seq_to_user[sequence_idx]
                    elif user_idx is not None:
                        try:
                            u_idx_val = int(user_idx[i].item()) if hasattr(user_idx[i], 'item') else int(user_idx[i])
                            if user_id_lookup is not None:
                                user_id = user_id_lookup[u_idx_val]
                            else:
                                # Fallback: use index as user ID if no lookup provided
                                user_id = u_idx_val
                        except Exception:
                            pass
                    
                    if user_id is not None:
                        user_predictions[user_id].append(float(pred_val))
                        user_actuals[user_id].append(float(actual_val))

                    sequence_idx += 1
        
        # Calculate per-user metrics
        per_user_metrics = {}
        for user_id in sorted(user_predictions.keys()):
            preds = np.array(user_predictions[user_id])
            actuals = np.array(user_actuals[user_id])
            
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mae = mean_absolute_error(actuals, preds)
            r2 = r2_score(actuals, preds)
            mse = mean_squared_error(actuals, preds)
            
            per_user_metrics[user_id] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'n_sequences': len(preds),
                'avg_efficiency': float(np.mean(actuals)),
                'std_efficiency': float(np.std(actuals)),
                'min_efficiency': float(np.min(actuals)),
                'max_efficiency': float(np.max(actuals)),
                'avg_prediction': float(np.mean(preds)),
                'std_prediction': float(np.std(preds))
            }
            
            logger.info(f'\nUser {user_id}:')
            logger.info(f'  Sequences: {len(preds)}')
            logger.info(f'  RMSE: {rmse:.4f} | R²: {r2:.4f}')
            logger.info(f'  Actual efficiency: {np.mean(actuals):.3f}±{np.std(actuals):.3f}')
        
        # Aggregate metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        aggregated_metrics = {
            'mse': float(mean_squared_error(all_actuals, all_predictions)),
            'rmse': float(np.sqrt(mean_squared_error(all_actuals, all_predictions))),
            'mae': float(mean_absolute_error(all_actuals, all_predictions)),
            'r2': float(r2_score(all_actuals, all_predictions)),
            'n_sequences': len(all_predictions),
            'n_users': len(user_predictions)
        }
        
        logger.info(f'\n--- AGGREGATED RESULTS ---')
        logger.info(f'Sequences: {aggregated_metrics["n_sequences"]} from {aggregated_metrics["n_users"]} users')
        logger.info(f'RMSE: {aggregated_metrics["rmse"]:.4f} | R²: {aggregated_metrics["r2"]:.4f}')
        
        return per_user_metrics, aggregated_metrics, all_predictions, all_actuals, dict(user_predictions), dict(user_actuals)
    



def main():
    """Run the shared test pipeline."""
    from test_model import main as test_main

    return test_main()


if __name__ == '__main__':
    main()
