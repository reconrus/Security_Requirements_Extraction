import logging
import os
import re
from argparse import ArgumentParser
from typing import Optional, Union

import pandas as pd
import torch
from streamlit import progress
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
)
from tqdm import tqdm

from dataset import (
    prepare_labels_mappings, read_dataframe, 
    write_dataframe, SecReqDataset,
)
from constants import (
    MAX_LENGTH, PREDICTING_APPLICATION_NAME, MODEL_TYPE,
    TMP_FOLDER_NAME, MODEL_FILENAME, PREDICT_DATASET_PATH,
)
from train import load_model

logger = logging.getLogger(PREDICTING_APPLICATION_NAME)

def setup_parser(parser):
    parser.add_argument(
        "-p", "--predict_dataset",
        help="path to predict dataset",
    )
    parser.add_argument(
        "-m", "--model_name",
        help="model name",
        default=MODEL_FILENAME,
    )
    parser.add_argument(
        "-o", "--output_path",
        help="path to output",
        default=MODEL_FILENAME,
    )
    parser.add_argument(
        "-l", "--max_len",
        help="maximum input sequence length",
        default=MAX_LENGTH,
    ),
    parser.add_argument(
        "-m", "--model_type",
        help="T5 model version (e.g. t5-small)",
        default=MODEL_TYPE,
    )


def prepare_data(dataframe, model_type, max_len):
    logger.info("===Started tokenizer loading===")
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    logger.info("===Finished tokenizer loading===")

    logger.info("===Started data preparation===")    
    dataset = SecReqDataset(dataframe, tokenizer, False, max_len)

    if not os.path.isdir(TMP_FOLDER_NAME):
        os.mkdir(TMP_FOLDER_NAME)
    torch.save(dataset, PREDICT_DATASET_PATH)

    prepare_labels_mappings(tokenizer)
    logger.info("===Finished data preparation===")
    return tokenizer


def predict(model: Union[T5ForConditionalGeneration, str], 
            dataframe: Optional[pd.DataFrame] = None,
            tokenizer: T5Tokenizer = None,
            streamlit_bar: bool = False):
    model = load_model(model) if isinstance(model, str) else model
    if dataframe is None:
        dataset = torch.load(PREDICT_DATASET_PATH)
    else:
        dataset = SecReqDataset(dataframe, tokenizer, train=False) 
    loader = DataLoader(dataset, batch_size=4)
    
    outputs = []
    
    if streamlit_bar:
        st_bar = progress(0)

    for i, batch in enumerate(tqdm(iter(loader), total=len(loader))):
        outs = model.generate(
            batch['input_ids'].to(model.device), 
            attention_mask=batch['attention_mask'].to(model.device)
        )
        outputs.extend(outs)

        if streamlit_bar:
            st_bar.progress(i / (len(loader) - 1))

    return outputs


def process_predictions(tokenizer, predictions, dataframe):
    predicted_labels = [tokenizer.decode(ids) for ids in predictions]
    processed_labels = [re.sub("<pad>|</s>", "", label).strip() for label in predicted_labels]
    
    dataframe["Label"] = processed_labels
    return dataframe


def main():
    parser = ArgumentParser(prog=PREDICTING_APPLICATION_NAME)
    setup_parser(parser)
    arguments = parser.parse_args()

    dataframe = read_dataframe(arguments.predict_dataset)
    tokenizer = prepare_data(dataframe, arguments.model_type, arguments.max_len)
    predictions = predict(arguments.model_name)
    
    dataframe = process_predictions(tokenizer, predictions, dataframe)
    write_dataframe(dataframe, arguments.output_path)


if __name__=="__main__":
    main()