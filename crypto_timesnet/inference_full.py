"""
Run inference on the full dataset using the trained best_model.pth and save predictions to CSV.
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config
from data_processor import DataProcessor, CryptoDataset
from model import TimesNetModel
from utils import load_checkpoint


def build_full_dataset(processor, config):
    """Load data, build features, and return train/full DataFrames plus feature list."""
    train_df, val_df, test_df = processor.load_and_split()
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feature_cols = processor.engineer.get_feature_columns()
    return train_df, full_df, feature_cols


def run_inference():
    config = get_config()
    device = torch.device(config.train.device)

    processor = DataProcessor(config)
    train_df, full_df, feature_cols = build_full_dataset(processor, config)

    # Keep enc_in consistent with training-time feature count
    config.model.enc_in = len(feature_cols)

    seq_len = config.data.seq_len

    # Fit scaler on train split to keep distribution consistent with training
    train_dataset = CryptoDataset(train_df, seq_len, feature_cols, mode='train')
    full_dataset = CryptoDataset(full_df, seq_len, feature_cols, scaler=train_dataset.scaler, mode='test')

    full_loader = DataLoader(
        full_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    # Prepare model
    model = TimesNetModel(config.model).to(device)
    checkpoint_path = os.path.join(config.train.save_dir, 'best_model.pth')
    load_checkpoint(checkpoint_path, model)
    model.eval()

    times = full_df[config.data.time_col].iloc[seq_len - 1 :].reset_index(drop=True)
    true_labels = full_df[config.data.label_col].iloc[seq_len - 1 :].reset_index(drop=True)

    rows = []
    pred_list = []
    cursor = 0
    with torch.no_grad():
        for batch in full_loader:
            x_enc = batch['x_enc'].to(device)
            x_mark_enc = batch['x_mark_enc'].to(device)

            outputs = model(x_enc, x_mark_enc)
            probs = F.softmax(outputs, dim=1).cpu()
            preds = probs.argmax(dim=1)

            batch_size = probs.shape[0]
            batch_times = times.iloc[cursor : cursor + batch_size].values
            batch_true = true_labels.iloc[cursor : cursor + batch_size].values
            cursor += batch_size

            for t, y_true, pred, p in zip(batch_times, batch_true, preds, probs):
                pred_list.append(int(pred.item()))
                rows.append({
                    'time': t,
                    'label': int(y_true),
                    'pred': int(pred.item()),
                    'prob_0': float(p[0].item()),
                    'prob_1': float(p[1].item()),
                    'prob_2': float(p[2].item()),
                })

    out_df = pd.DataFrame(rows)
    os.makedirs(config.train.save_dir, exist_ok=True)
    out_path = os.path.join(config.train.save_dir, 'inference_full.csv')
    out_df.to_csv(out_path, index=False)

    # Attach predictions back to the original full data (first seq_len-1 rows have no prediction)
    full_with_pred = full_df.copy()
    full_with_pred['pred'] = pd.NA
    start_idx = seq_len - 1
    end_idx = start_idx + len(pred_list)
    full_with_pred.loc[start_idx:end_idx - 1, 'pred'] = pred_list
    full_out_path = os.path.join(config.train.save_dir, 'full_with_pred.csv')
    full_with_pred.to_csv(full_out_path, index=False)

    print(f"Inference finished. Saved detailed probs to {out_path}")
    print(f"Full data with predictions saved to {full_out_path}")


if __name__ == '__main__':
    run_inference()
