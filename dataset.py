import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import (
  MAX_LENGTH, SEC_IDX, NON_SEC_IDX,
  SEC_LABEL, NONSEC_LABEL, 
  COLUMNS, DOCUMENT_COLUMN,
)

class SecReqDataset(Dataset):
  def __init__(self, original_dataframe, tokenizer, train=True, max_len=MAX_LENGTH):
    self.tokenizer = tokenizer
    self.train = train
    self.max_len = max_len
    self._load_dataset(original_dataframe)

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.data.items()}
    return item

  def __len__(self):
    return len(self.data["input_ids"])

  def _load_dataset(self, dataframe):
    dataset_df = self._format2t5(dataframe)
    self.data = self._convert_to_features(dataset_df)

  def _format2t5(self, df): 
    def _process(x):
      input = f"security classification: {x[0]}"
      if self.train:
        return pd.Series({"inputs": input, "targets": x[1]})
      else: 
        return pd.Series({"inputs": input})
    return df.apply(_process, axis=1)

  def _convert_to_features(self, df):
      if self.train:
        encodings = self.tokenizer.prepare_seq2seq_batch(
            df.inputs.to_list(),
            df.targets.to_list(),
            max_length=self.max_len,
            return_tensors="pt",
            )
        encodings = {
            'input_ids': encodings['input_ids'], 
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['labels'],
        }
      else:
        encodings = self.tokenizer(
            df.inputs.to_list(), 
            padding=True,
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
            )
        encodings = {
            'input_ids': encodings['input_ids'], 
            'attention_mask': encodings['attention_mask'],
        }

      return encodings


def read_dataframe(path):
    dataset = pd.read_csv(path, sep="\t")
    return dataset


def write_dataframe(df, path):
    df.to_csv(path, sep="\t")


def read_documents(dataset_path, document_names):
    documents = pd.DataFrame(columns=COLUMNS + [DOCUMENT_COLUMN])
    for document_name in document_names:
        document_path = os.path.join(dataset_path, f"{document_name}.csv")
        document_df = read_dataframe(document_path)
        document_df[DOCUMENT_COLUMN] = document_name
        documents = documents.append(document_df)
    return documents


def read_data(datasets_folder, datasets_paths):
    resulting_dataframe = pd.DataFrame(columns=COLUMNS + [DOCUMENT_COLUMN])
    for dataset, document_names in datasets_paths.items():
        datasets_path = os.path.join(datasets_folder, dataset)
        documents_df = read_documents(datasets_path, document_names)
        resulting_dataframe = resulting_dataframe.append(documents_df)
    return resulting_dataframe


idxs_to_label = {}

def prepare_labels_mappings(tokenizer):
    global idxs_to_label
    labels = [SEC_LABEL, NONSEC_LABEL]
    sec_idxs, non_sec_idxs = tokenizer.prepare_seq2seq_batch(labels)['input_ids']
    idxs_to_label[tuple(sec_idxs)] = SEC_IDX
    idxs_to_label[tuple(non_sec_idxs)] = NON_SEC_IDX
