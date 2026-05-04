"""Quick smoke test to validate data pipeline and model forward passes.

This script attempts to load a small portion of the CSV at `DATA_PATH` and
falls back to a synthetic example if the CSV is missing or too small.
It then instantiates both models and runs a forward pass to catch shape/type issues.
"""
from pathlib import Path
import numpy as np
import torch

from config import DATA_PATH, SEQUENCE_LENGTH, USER_EMB_DIM
from data_preprocessor import SleepDataPreprocessor
from model import LSTMModel
from model_without_att import LSTMModelNoAttention


def build_synthetic_df(num_users=3, days=10):
    import pandas as pd
    rows = []
    for u in range(num_users):
        for d in range(days):
            rows.append({
                'userId': f'user_{u}',
                'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=d),
                'go2bed': f"{22 + (d % 2)}:30:00",
                'sleep_duration': float(7 + (d % 3) * 0.2),
                'sleep_latency': float(0.3 + (d % 2) * 0.1),
                'waso': float((d % 2) * 5),
                'wakeup@night': int(d % 2),
                'sleep_efficiency': float(0.8 + (d % 4) * 0.02),
            })
    return pd.DataFrame(rows)


def main():
    device = torch.device('cpu')
    pre = SleepDataPreprocessor(sequence_length=SEQUENCE_LENGTH)

    data_path = Path(DATA_PATH)
    if data_path.exists():
        try:
            X, y, metadata = pre.load_and_prepare(str(data_path))
        except Exception as e:
            print('Failed to load real CSV, falling back to synthetic:', e)
            df = build_synthetic_df()
            tmp = Path('data/tmp_smoke.csv')
            tmp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(tmp, index=False)
            X, y, metadata = pre.load_and_prepare(str(tmp))
    else:
        print('Data file not found; using synthetic dataset')
        df = build_synthetic_df()
        tmp = Path('data/tmp_smoke.csv')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=False)
        X, y, metadata = pre.load_and_prepare(str(tmp))

    print('Shapes:', X.shape, y.shape)

    # Select a small batch
    batch_X = torch.FloatTensor(X[:4])
    batch_user_idx = None
    if 'seq_user_idx' in metadata:
        import numpy as np
        seq_user_idx = metadata['seq_user_idx']
        batch_user_idx = torch.LongTensor(seq_user_idx[:4])

    input_size = X.shape[2]
    num_users = len(metadata['users']) if 'users' in metadata else 0

    model1 = LSTMModel(input_size=input_size, hidden_size=16, num_layers=1, output_size=1, num_users=num_users, user_emb_dim=USER_EMB_DIM)
    model2 = LSTMModelNoAttention(input_size=input_size, hidden_size=16, num_layers=1, output_size=1, num_users=num_users, user_emb_dim=USER_EMB_DIM)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(batch_X, batch_user_idx) if batch_user_idx is not None else model1(batch_X)
        out2 = model2(batch_X, batch_user_idx) if batch_user_idx is not None else model2(batch_X)

    print('Model outputs shapes:', out1.shape, out2.shape)
    print('Smoke test passed')


if __name__ == '__main__':
    main()
