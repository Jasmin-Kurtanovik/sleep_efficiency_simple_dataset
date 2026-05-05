"""
Utility functions for data loading and preprocessing.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data.
    
    Args:
        data (np.ndarray): Time series data of shape (n_samples, n_features)
        sequence_length (int): Length of input sequences
        target_length (int): Length of target sequences (default: 1)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        target_length: int = 1
    ):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.target_length = target_length
    
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.target_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.target_length, 0]
        return x, y


def create_data_loaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    sequence_length: int,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        sequence_length: Length of input sequences
        batch_size: Batch size for loaders
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def normalize_data(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Normalize data using MinMaxScaler.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        
    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_data_norm = scaler.fit_transform(train_data)
    val_data_norm = scaler.transform(val_data)
    test_data_norm = scaler.transform(test_data)
    
    return train_data_norm, val_data_norm, test_data_norm, scaler


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
