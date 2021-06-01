import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support
)

from constants import SEC_IDX
from dataset import LabelsData


def f1_score_with_invalid(targets, predictions):
    """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.
    Args:
        targets: np.ndarray of targets, either 0 or 1
        predictions: np.ndarray of predictions, any integer value
    Returns:
        F1 score, where any prediction != 0 or 1 is counted as wrong.
    Source: https://github.com/google-research/text-to-text-transfer-transformer/blob/6e7593970af8a3b221ceb6a9eec09d48377e47ac/t5/evaluation/metrics.py#L232
    """
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    # For any prediction != 0 or 1, set it to the opposite of what the target is
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * f1_score(targets, predictions)}


def append_metrics_to_file(metrics, metrics_folder, file_name):
    metrics_df = pd.DataFrame({key: [value] for key, value in metrics.items()})
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)
    file_path = os.path.join(metrics_folder, file_name)
    metrics_df.to_csv(file_path, mode="a", header=False)


def compute_metrics(pred, labels_data: LabelsData, invalid_to_sec=False):
    """
    :param invalid_to_sec: map invalid prediction to security or not.
        True assumes that it is better to have excess non-security requirements
        labeled as security than miss any security variable
    """
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    def _convert_to_labels(idxs):
        label_to_set = SEC_IDX if invalid_to_sec else -1
        label = labels_data.idxs_to_label.get(tuple(idxs), label_to_set)
        return label

    targets = np.fromiter(map(_convert_to_labels, labels), dtype=np.int32)
    predictions  = np.fromiter(map(_convert_to_labels, preds), dtype=np.int32)

    invalid_idx_mask = predictions == -1
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]

    acc = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }