import logging
import os
import yaml
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments,
)

from dataset import (
    idxs_to_label, prepare_labels_mappings,
    read_data, read_dataframe, SecReqDataset,
)
from callback import EarlyStoppingCallback
from constants import (
    SEC_LABEL, NONSEC_LABEL, SEC_IDX, NON_SEC_IDX, COLUMNS,
    TRAINING_APPLICATION_NAME, TMP_FOLDER_NAME, YAML_CONFIG_PATH,
    TRAIN_DATASET_PATH, VALID_DATASET_PATH, MODEL_FOLDER,
)
from metrics import append_metrics_to_file


logger = logging.getLogger(TRAINING_APPLICATION_NAME)


def compute_metrics(pred, invalid_to_sec=False):
    """
    :param invalid_to_sec: map invalid prediction to security or not.
        True assumes that it is better to have excess non-security requirements
        labeled as security than miss any security variable
    """
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    def _convert_to_labels(idxs):
        label_to_set = SEC_IDX if invalid_to_sec else -1
        label = idxs_to_label.get(tuple(idxs), label_to_set)
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


def load_model(model_path, device="cuda"):
    logger.info("===Started model loading===")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    logger.info("===Finished model loading===")
    return model


def train(model_type, epochs):
    model = load_model(model_type)

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        warmup_steps=300,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        load_best_model_at_end = True,
        metric_for_best_model='f1',
        output_dir='models',
    )

    train_dataset  = torch.load(TRAIN_DATASET_PATH)
    valid_dataset = torch.load(VALID_DATASET_PATH)

    early_stopping_callback = EarlyStoppingCallback()

    logger.info("===Started model training===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    logger.info("===Finished model training===")

    evaluation_with_invalid = trainer.evaluate()
    trainer.compute_metrics = lambda x: compute_metrics(x, invalid_to_sec=True)
    evalutaion_with_sec = trainer.evaluate()
    return [evaluation_with_invalid, evalutaion_with_sec]


def prepare_data(train_dataframe, valid_dataframe, model_type, max_len):
    logger.info("===Started tokenizer loading===")
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    logger.info("===Finished tokenizer loading===")

    logger.info("===Started data preparation===")
    train_dataset = SecReqDataset(train_dataframe, tokenizer, True, max_len)
    valid_dataset = SecReqDataset(valid_dataframe, tokenizer, True, max_len)

    if not os.path.isdir(TMP_FOLDER_NAME):
        os.mkdir(TMP_FOLDER_NAME)
    torch.save(train_dataset, TRAIN_DATASET_PATH)
    torch.save(valid_dataset, VALID_DATASET_PATH)

    prepare_labels_mappings(tokenizer)
    logger.info("===Finished data preparation===")


def cross_evaluation(model_type, full_train, epochs, max_len):
    skf = StratifiedKFold(n_splits=10)

    metrics_with_invalid = defaultdict(list)
    metrics_with_sec = defaultdict(list)

    for train_index, valid_index in skf.split(full_train["Text"], full_train["Label"]):
        train_df = full_train.iloc[train_index]
        valid_df = full_train.iloc[valid_index]
        prepare_data(train_df, valid_df, model_type, max_len)

        evaluation = train(model_type, epochs)
        for key, value in evaluation[0].items():
            metrics_with_invalid[key].append(value)
        for key, value in evaluation[1].items():
            metrics_with_sec[key].append(value)

        append_metrics_to_file(evaluation[0], f"reverse_{metrics_file_path}")
        append_metrics_to_file(evaluation[1], f"security_{metrics_file_path}")

    mean_metrics_invalid = {key: np.mean(value) for key, value in metrics_with_invalid.items()}
    metrics_with_sec = {key: np.mean(value) for key, value in metrics_with_sec.items()}
    return [mean_metrics_invalid, metrics_with_sec]


def train_and_evaluate(model_type, train_dataframe, valid_dataframe,
                       epochs, max_len, validation_type, metrics_file_path):

    if validation_type == "p-validation":
        prepare_data(train_dataframe, valid_dataframe, model_type, max_len)
        metrics = train(model_type, epochs)
        append_metrics_to_file(metrics[0], f"reverse_{metrics_file_path}")
        append_metrics_to_file(metrics[1], f"security_{metrics_file_path}")
    elif validation_type == "cross-validation":
        metrics = cross_evaluation(model_type, train_dataframe, epochs, max_len)
    else:
        logger.exception("Unsupported validation method")

    print("Evaluation results: \n")
    print("Invalid predictions set as incorrect: ", metrics[0])
    print("Invalid predictions set as security: ", metrics[1])


if __name__=="__main__":
    with open(YAML_CONFIG_PATH, "r") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        training_parameters = yaml.load(file, Loader=yaml.FullLoader)
    datasets_path = training_parameters["datasets_path"]
    train_datasets = training_parameters["train_datasets"]
    valid_datasets = training_parameters["valid_datasets"]
    train_dataframe = read_data(datasets_path, train_datasets)
    valid_dataframe = read_data(datasets_path, valid_datasets)
    metrics_file_path = training_parameters["metrics_file"]
    train_and_evaluate(model_type=training_parameters["model_type"],
                       train_dataframe=train_dataframe,
                       valid_dataframe=valid_dataframe,
                       epochs=training_parameters["epochs"],
                       max_len=training_parameters["max_len"],
                       validation_type=training_parameters["validation"],
                       metrics_file_path=metrics_file_path,
                       )