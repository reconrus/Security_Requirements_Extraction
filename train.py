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
    Trainer, TrainingArguments
)

from dataset import (
    idxs_to_label, prepare_labels_mappings, read_data, read_dataframe, SecReqDataset,
)
from constants import (
    SEC_LABEL, NONSEC_LABEL, SEC_IDX, NON_SEC_IDX, COLUMNS,
    TRAINING_APPLICATION_NAME, TMP_FOLDER_NAME, YAML_CONFIG_PATH,
    TRAIN_DATASET_PATH, VALID_DATASET_PATH, MODEL_FOLDER,
)
from metrics import f1_score_with_invalid


logger = logging.getLogger(TRAINING_APPLICATION_NAME)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    def _convert_to_labels(idxs):
    #   label = idxs_to_label.get(tuple(idxs), -1)
    # assuming that it is better to have excess non-security requirements
    # labeled as security than miss any security variable
        label = idxs_to_label.get(tuple(idxs), SEC_IDX)
        return label

    targets = np.fromiter(map(_convert_to_labels, labels), dtype=np.int)
    predictions  = np.fromiter(map(_convert_to_labels, preds), dtype=np.int)
    wrong_predictions = np.where((predictions == -1))[0]
    wrong_predictions_number = wrong_predictions.shape[0]

    acc = accuracy_score(targets, predictions)
    f1 = f1_score_with_invalid(targets, predictions)

    targets = np.delete(targets, wrong_predictions)
    predictions = np.delete(predictions, wrong_predictions)
    precision, recall, _, _ = precision_recall_fscore_support(targets, predictions, average='binary')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'wrong_predictions': wrong_predictions_number,
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
    )

    train_dataset  = torch.load(TRAIN_DATASET_PATH)
    valid_dataset = torch.load(VALID_DATASET_PATH)

    logger.info("===Started model training===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    logger.info("===Finished model training===")

    evaluation = trainer.evaluate()
    return evaluation


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

    metrics = defaultdict(list)

    for train_index, valid_index in skf.split(full_train["Text"], full_train["Label"]):
        train_df = full_train.iloc[train_index]
        valid_df = full_train.iloc[valid_index]
        prepare_data(train_df, valid_df, model_type, max_len)

        evaluation = train(model_type, epochs)
        for key, value in evaluation.items():
            metrics[key].append(value)

    mean_metrics = {key: np.mean(value) for key, value in metrics.items()}
    return mean_metrics


def train_and_evaluate(model_type, train_dataframe, valid_dataframe,
                       epochs, max_len, validation_type):

    if validation_type == "p-validation":
        prepare_data(train_dataframe, valid_dataframe, model_type, max_len)
        metrics = train(model_type, epochs)
    elif validation_type == "cross-validation":
        metrics = cross_evaluation(model_type, train_dataframe, epochs, max_len)
    else:
        logger.exception("Unsupported validation method")

    print("Evaluation results: ", metrics)


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
    train_and_evaluate(model_type=training_parameters["model_type"],
                       train_dataframe=train_dataframe,
                       valid_dataframe=valid_dataframe,
                       epochs=training_parameters["epochs"],
                       max_len=training_parameters["max_len"],
                       validation_type=training_parameters["validation"],
                       )