import logging
import os
import random
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments,
)

from dataset import (
    oversample_dataset, LabelsData, SecReqDataset,
)
from callback import EarlyStoppingCallback
from configuration import TrainConfiguration
from constants import (
    TRAINING_APPLICATION_NAME, TMP_FOLDER_NAME,
    TRAIN_DATASET_PATH, VALID_DATASET_PATH, MODEL_FOLDER,
)
from metrics import append_metrics_to_file, compute_metrics


logger = logging.getLogger(TRAINING_APPLICATION_NAME)


def clear_models_folder():
    shutil.rmtree(MODEL_FOLDER)
    os.mkdir(MODEL_FOLDER)


def load_model(model_path: str, device: str = "cuda"):
    logger.info("===Started model loading===")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    logger.info("===Finished model loading===")
    return model


def train(model_type: str, epochs: int, labels_data: LabelsData, seed: int):
    model = load_model(model_type)

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        warmup_steps=300,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        metric_for_best_model='f1',
        output_dir=MODEL_FOLDER,
        load_best_model_at_end=True,
        seed=seed,
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
        compute_metrics=lambda x: compute_metrics(x, labels_data),
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    logger.info("===Finished model training===")

    evaluation_with_invalid = trainer.evaluate()
    trainer.compute_metrics = lambda x: compute_metrics(x, labels_data, invalid_to_sec=True)
    evalutaion_with_sec = trainer.evaluate()
    return [evaluation_with_invalid, evalutaion_with_sec]


def prepare_data(train_dataframe: pd.DataFrame,
                 valid_dataframe: pd.DataFrame,
                 model_type: str, max_len: int,
                 oversampling: bool, labels_data: LabelsData):
    logger.info("===Started tokenizer loading===")
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    logger.info("===Finished tokenizer loading===")

    logger.info("===Started data preparation===")
    if oversampling:
        train_dataframe = oversample_dataset(train_dataframe, labels_data)
    train_dataset = SecReqDataset(train_dataframe, tokenizer, True, max_len)
    valid_dataset = SecReqDataset(valid_dataframe, tokenizer, True, max_len)

    if not os.path.isdir(TMP_FOLDER_NAME):
        os.mkdir(TMP_FOLDER_NAME)
    torch.save(train_dataset, TRAIN_DATASET_PATH)
    torch.save(valid_dataset, VALID_DATASET_PATH)

    labels_data.prepare_labels_mappings(tokenizer)
    logger.info("===Finished data preparation===")


def cross_evaluation(model_type: str, full_train: pd.DataFrame,
                     epochs: int, max_len: int, oversampling: bool,
                     clear_models_dir: bool, labels_data: LabelsData,
                     metrics_folder: str, metrics_file_path: str, seed: int):
    skf = StratifiedKFold(n_splits=10)

    metrics_with_invalid = defaultdict(list)
    metrics_with_sec = defaultdict(list)

    for train_index, valid_index in skf.split(full_train["Text"], full_train["Label"]):
        train_df = full_train.iloc[train_index]
        valid_df = full_train.iloc[valid_index]
        prepare_data(train_df, valid_df, model_type, max_len, oversampling, labels_data)

        evaluation = train(model_type, epochs, labels_data, seed)
        for key, value in evaluation[0].items():
            metrics_with_invalid[key].append(value)
        for key, value in evaluation[1].items():
            metrics_with_sec[key].append(value)

        append_metrics_to_file(evaluation[0], metrics_folder, f"reverse_{metrics_file_path}")
        append_metrics_to_file(evaluation[1], metrics_folder, f"security_{metrics_file_path}")
        if clear_models_dir:
            clear_models_folder()

    mean_metrics_invalid = {key: np.mean(value) for key, value in metrics_with_invalid.items()}
    metrics_with_sec = {key: np.mean(value) for key, value in metrics_with_sec.items()}
    return [mean_metrics_invalid, metrics_with_sec]


def train_and_evaluate(model_type: str, train_dataframe: pd.DataFrame,
                       valid_dataframe: pd.DataFrame, epochs: int,
                       max_len: int, validation_type: str,
                       metrics_folder: str, metrics_file_path: str,
                       oversampling: bool, clear_models_dir: bool, 
                       labels_data: LabelsData, seed: int):
    if validation_type == "p-validation":
        prepare_data(train_dataframe, valid_dataframe, model_type, max_len, False, labels_data)
        metrics = train(model_type, epochs, labels_data, seed)
        append_metrics_to_file(metrics[0], metrics_folder, f"reverse_{metrics_file_path}")
        append_metrics_to_file(metrics[1], metrics_folder, f"security_{metrics_file_path}")
    elif validation_type == "cross-validation":
        metrics = cross_evaluation(model_type=model_type,
                                   full_train=train_dataframe,
                                   epochs=epochs,
                                   max_len=max_len,
                                   oversampling=oversampling,
                                   clear_models_dir=clear_models_dir,
                                   labels_data=labels_data,
                                   metrics_folder=metrics_folder,
                                   metrics_file_path=metrics_file_path,
                                   seed=seed)
    else:
        logger.exception("Unsupported validation method")

    if clear_models_dir:
        clear_models_folder()

    print("Evaluation results: \n")
    print("Invalid predictions set as incorrect: ", metrics[0])
    print("Invalid predictions set as security: ", metrics[1])


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


if __name__=="__main__":
    config_path = sys.argv[1]
    configuration = TrainConfiguration.from_yaml(config_path)
    set_seeds(configuration.seed)
    train_and_evaluate(model_type=configuration.model_type,
                       train_dataframe=configuration.datasets.train_dataframe,
                       valid_dataframe=configuration.datasets.valid_dataframe,
                       epochs=configuration.epochs,
                       max_len=configuration.max_len,
                       validation_type=configuration.validation_type,
                       metrics_folder=configuration.metrics_folder,
                       metrics_file_path=configuration.metrics_file_path,
                       oversampling=configuration.oversampling,
                       clear_models_dir=configuration.clear_models_dir,
                       labels_data=configuration.labels_data,
                       seed=configuration.seed,
                       )
