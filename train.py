import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments,
)

from dataset import read_dataframe, SecReqDataset
from constants import (
    MAX_LENGTH, MODEL_TYPE, DEFAULT_EPOCHS,
    SEC_LABEL, NONSEC_LABEL,
    TRAINING_APPLICATION_NAME,
    TMP_FOLDER_NAME, MODEL_FOLDER, MODEL_FILENAME,
    TRAIN_DATASET_PATH, VALID_DATASET_PATH,
)

logger = logging.getLogger(TRAINING_APPLICATION_NAME)


def setup_parser(parser):
    parser.add_argument(
        "-d", "--train_path",
        help="path to train dataset",
    )
    parser.add_argument(
        "-v", "--valid_path",
        help="path to valid dataset",
    )
    parser.add_argument(
        "-o", "--output_model_name",
        help="model output name",
        default=MODEL_FILENAME,
    )
    parser.add_argument(
        "-l", "--max_len",
        help="maximum input sequence length"
        default=MAX_LENGTH,
    )
    parser.add_argument(
        "-m", "--model_type",
        help="T5 model version (e.g. t5-small)",
        default=MODEL_TYPE,
    )
    parser.add_argument(
        "-e", "--epochs",
        help="number of epochs to train model",
        default=DEFAULT_EPOCHS,
    )

def prepare_labels_mappings(tokenizer):
    global idxs_to_label
    labels = [SEC_LABEL, NONSEC_LABEL]
    sec_idxs, non_sec_idxs = tokenizer.prepare_seq2seq_batch(labels)['input_ids']
    idxs_to_label = {
        tuple(sec_idxs): 1,
        tuple(non_sec_idxs): 0,
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    def _convert_to_labels(idxs):
      label = idxs_to_label.get(tuple(idxs), -1)
      return label

    targets = np.fromiter(map(_convert_to_labels, labels), dtype=np.int)
    predictions  = np.fromiter(map(_convert_to_labels, preds), dtype=np.int)
    wrong_predictions = np.where((predictions == -1))[0]
    wrong_predictions_number = wrong_predictions.shape[0]

    acc = accuracy_score(targets, predictions)
    targets = np.delete(targets, wrong_predictions)
    predictions = np.delete(predictions, wrong_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')

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


def train(epochs):
    model = load_model(arguments.model_type)
    
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

    return model


def prepare_data(train_path, valid_path, model_type, max_len):
    logger.info("===Started tokenizer loading===")
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    logger.info("===Finished tokenizer loading===")

    logger.info("===Started data preparation===")
    train_dataframe = read_dataframe(train_path)
    valid_dataframe = read_dataframe(valid_path)
    
    train_dataset = SecReqDataset(train_dataframe, tokenizer, True, max_len)
    valid_dataset = SecReqDataset(valid_dataframe, tokenizer, True, max_len)

    if not os.path.isdir(TMP_FOLDER_NAME):
        os.mkdir(TMP_FOLDER_NAME)
    torch.save(train_dataset, TRAIN_DATASET_PATH)
    torch.save(valid_dataset, VALID_DATASET_PATH)

    prepare_labels_mappings(tokenizer)
    logger.info("===Finished data preparation===")


def main():
    parser = ArgumentParser(prog=TRAINING_APPLICATION_NAME)
    setup_parser(parser)
    arguments = parser.parse_args()

    prepare_data(arguments.train_path, arguments.valid_path, 
                 arguments.model_type, arguments.max_len)
    model = train(arguments.epochs)

    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    model.save_pretrained(os.path.join(MODEL_FOLDER, arguments.output_model_name))


if __name__=="__main__":
    main()