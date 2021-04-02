import numpy as np
import pandas as pd
import sklearn.metrics


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
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


def append_metrics_to_file(metrics, file_path):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(file_path, mode="a", header=False)
