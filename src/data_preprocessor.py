"""
Data preprocessing for LSTM training on sleep diary data.
Handles temporal sequencing, feature engineering, and periodicity encoding.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Tuple, List, Dict
import warnings

from config import LOOKBACK_DAYS, RANDOM_SEED

warnings.filterwarnings('ignore')


class SleepDataPreprocessor:
    """
    Preprocesses sleep diary data for LSTM training.
    
    Key features:
    - Extracts time-based features (hour, day of week) for periodicity
    - Creates sequences to capture temporal patterns
    - Handles per-user data (no leakage between users)
    - Normalizes features appropriately
    """
    
    def __init__(self, sequence_length: int = LOOKBACK_DAYS):
        """
        Args:
            sequence_length: Number of days to look back (default: LOOKBACK_DAYS)
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_prepare(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load CSV and prepare data for LSTM.
        
        Returns:
            Tuple of (X, y, metadata) where:
            - X: shape (N_samples, sequence_length, n_features)
            - y: shape (N_samples,) - target values
            - metadata: user info and indices
        """
        print("Loading and preprocessing sleep diary data...")
        
        # Load data
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])

        # Create feature columns
        print(f"Original data shape: {df.shape}")

        # Extract time features for periodicity
        df['dayofweek'] = df['date'].dt.dayofweek  # 0-6 (Monday-Sunday)

        # Robust parsing for bedtime hour (accepts multiple time formats)
        if 'go2bed' in df.columns:
            try:
                # Try parsing with explicit %H:%M:%S format first
                bedtime_series = pd.to_datetime(df['go2bed'], format='%H:%M:%S', errors='coerce').dt.hour
                # If that fails, try %H:%M format
                if bedtime_series.isna().all():
                    bedtime_series = pd.to_datetime(df['go2bed'], format='%H:%M', errors='coerce').dt.hour
                # Fallback: extract hour from string before ':'
                if bedtime_series.isna().any():
                    fallback = df['go2bed'].astype(str).str.split(':').str[0]
                    bedtime_series = bedtime_series.fillna(pd.to_numeric(fallback, errors='coerce'))
                df['bedtime_hour'] = bedtime_series.fillna(0).astype(int)
            except Exception:
                df['bedtime_hour'] = 0
        else:
            df['bedtime_hour'] = 0

        df['month_day'] = df['date'].dt.day
        
        # Sort by user and date to maintain temporal order
        df = df.sort_values(['userId', 'date']).reset_index(drop=True)
        
        print(f"\nFeatures extracted:")
        print(f"  - dayofweek: {df['dayofweek'].unique()}")
        print(f"  - bedtime_hour: {df['bedtime_hour'].min():.0f}-{df['bedtime_hour'].max():.0f}")
        
        # Select features for input
        # Detect column corresponding to waking at night (be permissive with names)
        wake_candidates = [c for c in df.columns if 'wakeup' in c.lower() or ('wake' in c.lower() and 'night' in c.lower())]
        if wake_candidates:
            wake_col = wake_candidates[0]
        else:
            # Try common alternatives
            alt_candidates = [c for c in df.columns if 'night' in c.lower() and 'wake' in c.lower()]
            wake_col = alt_candidates[0] if alt_candidates else 'wakeup@night'

        feature_cols = [
            'sleep_duration',      # Hours slept
            'sleep_latency',       # Time to fall asleep (hours)
            'waso',                # Wake after sleep onset (minutes)
            wake_col,              # Binary: woke up at night (flexible name)
            'dayofweek',           # 0-6 (captures weekly pattern)
            'bedtime_hour'         # Hour they went to bed (sleep schedule)
        ]
        
        # Handle missing values
        if 'waso' in df.columns:
            df['waso'] = df['waso'].fillna(0)
        if wake_col not in df.columns:
            df[wake_col] = 0
        else:
            df[wake_col] = df[wake_col].fillna(0)
        
        # Ensure all feature columns are numeric
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X_list = []
        y_list = []
        seq_user_idx = []
        metadata = {'users': [], 'user_start_idx': [], 'user_end_idx': []}
        
        # Process each user separately (important: no data leakage)
        users_unique = list(df['userId'].unique())
        for u_idx, user_id in enumerate(users_unique):
            user_data = df[df['userId'] == user_id].copy()
            user_start_idx = len(X_list)
            
            if len(user_data) < self.sequence_length + 1:
                print(f"  [WARNING] User {user_id}: only {len(user_data)} days (need {self.sequence_length + 1})")
                continue
            
            # Extract features and target
            X_user = user_data[feature_cols].values
            y_user = user_data['sleep_efficiency'].values
            
            # Normalize each user's data independently (better for personalization)
            # Reset scaler for each user to maintain independence
            user_scaler = MinMaxScaler(feature_range=(0, 1))
            X_user = user_scaler.fit_transform(X_user)
            
            # Create sequences: use past `sequence_length` days to predict next day
            for i in range(len(X_user) - self.sequence_length):
                X_seq = X_user[i:i + self.sequence_length]
                y_val = y_user[i + self.sequence_length]     # Target: next day's efficiency
                
                X_list.append(X_seq)
                y_list.append(y_val)
                seq_user_idx.append(u_idx)
            
            user_end_idx = len(X_list)
            metadata['users'].append(user_id)
            metadata['user_start_idx'].append(user_start_idx)
            metadata['user_end_idx'].append(user_end_idx)
            
            print(f"  [OK] User {user_id}: {user_end_idx - user_start_idx} sequences")
        
        X = np.array(X_list)
        y = np.array(y_list)
        seq_user_idx = np.array(seq_user_idx)
        
        print(f"\nSequences created:")
        print(f"  X shape: {X.shape} (samples, sequence_length, features)")
        print(f"  y shape: {y.shape}")
        print(f"  - Samples: {len(X)}")
        print(f"  - Sequence length: {self.sequence_length} days")
        print(f"  - Features per day: {X.shape[2]}")
        
        # Attach sequence->user mapping to metadata for downstream use
        metadata['seq_user_idx'] = seq_user_idx

        return X, y, metadata
    
    def get_feature_names(self) -> List[str]:
        """Get names of input features in order"""
        return [
            'sleep_duration',
            'sleep_latency',
            'waso',
            'wakeup@night',
            'dayofweek',
            'bedtime_hour'
        ]


class SleepDataset(Dataset):
    """PyTorch Dataset for sleep data sequences

    Optionally returns a per-sample integer `user_idx` when provided.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_user_idx: np.ndarray = None):
        """
        Args:
            X: Input sequences of shape (N, seq_len, features)
            y: Target values of shape (N,)
            seq_user_idx: Optional array of shape (N,) mapping each sequence to a user index
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_user_idx = torch.LongTensor(seq_user_idx) if seq_user_idx is not None else None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        if self.seq_user_idx is None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx], self.seq_user_idx[idx]


def split_data_by_user(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Tuple[np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray]]:
    """
    Split data while respecting user boundaries (no test data from training users).
    Uses RANDOM_SEED for reproducible splits (same train/val/test users across runs).
    
    Args:
        X, y: Full dataset
        metadata: User metadata
        train_ratio: Proportion of users for training
        val_ratio: Proportion of users for validation
        
    Returns:
        ((X_train, y_train, user_idx_train), (X_val, y_val, user_idx_val), (X_test, y_test, user_idx_test))
    """
    n_users = len(metadata['users'])
    n_train_users = int(n_users * train_ratio)
    n_val_users = int(n_users * val_ratio)
    
    # Set seed for reproducible user splits (ensures same users in train/val/test across runs)
    np.random.seed(RANDOM_SEED)
    user_indices = np.random.permutation(n_users)
    train_users = user_indices[:n_train_users]
    val_users = user_indices[n_train_users:n_train_users + n_val_users]
    test_users = user_indices[n_train_users + n_val_users:]
    
    def get_user_data(user_idxs):
        indices = []
        for u_idx in user_idxs:
            start = metadata['user_start_idx'][u_idx]
            end = metadata['user_end_idx'][u_idx]
            indices.extend(range(start, end))
        indices = np.array(indices)
        return X[indices], y[indices], metadata.get('seq_user_idx', None)[indices]
    
    X_train, y_train, user_idx_train = get_user_data(train_users)
    X_val, y_val, user_idx_val = get_user_data(val_users)
    X_test, y_test, user_idx_test = get_user_data(test_users)
    
    print(f"\nData split (by users, no leakage):")
    print(f"  Train: {n_train_users} users, {len(X_train)} sequences")
    print(f"  Val:   {n_val_users} users, {len(X_val)} sequences")
    print(f"  Test:  {len(test_users)} users, {len(X_test)} sequences")
    
    return (X_train, y_train, user_idx_train), (X_val, y_val, user_idx_val), (X_test, y_test, user_idx_test)


def main():
    """Example usage"""
    # Preprocess data
    preprocessor = SleepDataPreprocessor(sequence_length=LOOKBACK_DAYS)
    X, y, metadata = preprocessor.load_and_prepare('data/sleep_diary.csv')
    
    print("\n" + "="*80)
    print("FEATURE EXPLANATION:")
    print("="*80)
    print("""
LSTM Input Features (normalized 0-1):
  1. sleep_duration       - Total hours of sleep
  2. sleep_latency        - Hours to fall asleep (lower is better)
  3. waso                 - Minutes awake after sleep onset
  4. wakeup@night         - Binary: did you wake at night?
  5. dayofweek            - Day of week (0-6) → captures weekly pattern
  6. bedtime_hour         - Hour you went to bed → captures sleep schedule
  
PERIODICITY LEARNING:
    - Sequence length: LOOKBACK_DAYS = captures recent history
  - dayofweek feature: helps learn "I sleep better on weekends"
  - bedtime_hour: helps learn "I sleep better when I sleep at X time"
  - LSTM hidden state: learns longer dependencies

TARGET: sleep_efficiency (0-1)
  - What LSTM predicts: tomorrow's sleep efficiency
  - 0.9+ = excellent sleep, 0.8-0.9 = good, <0.8 = poor
    """)
    
    # Split data
    (X_train, y_train, user_idx_train), (X_val, y_val, user_idx_val), (X_test, y_test, user_idx_test) = split_data_by_user(
        X, y, metadata, train_ratio=0.6, val_ratio=0.2
    )
    
    # Create datasets (include per-sequence user index)
    train_dataset = SleepDataset(X_train, y_train, seq_user_idx=user_idx_train)
    val_dataset = SleepDataset(X_val, y_val, seq_user_idx=user_idx_val)
    test_dataset = SleepDataset(X_test, y_test, seq_user_idx=user_idx_test)
    
    print(f"\nDatasets ready for LSTM training!")
    print(f"  - Features per day: {preprocessor.get_feature_names()}")
    print(f"  - Predict 1 day ahead (next day's sleep efficiency)")


if __name__ == "__main__":
    main()
