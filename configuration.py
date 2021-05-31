from dataclasses import dataclass
from typing import List

import pandas as pd
import yaml

from dataset import read_data, LabelsData


@dataclass
class TrainConfiguration:
    model_type: str
    epochs: int
    max_len: int
    datasets_path: str
    train_datasets: List[str]
    valid_datasets: List[str]
    oversampling: bool
    clear_models_dir: bool
    early_stopping: bool
    validation_type: str
    train_dataframe: pd.DataFrame
    valid_dataframe: pd.DataFrame
    metrics_file_path: str
    labels_data: LabelsData

    @staticmethod
    def from_yaml(config_path: str):
        with open(config_path, "r") as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            training_parameters = yaml.load(file, Loader=yaml.FullLoader)
        
        model_type = training_parameters["model_type"]
        epochs = training_parameters["epochs"]
        max_len = training_parameters["max_len"]
        datasets_path = training_parameters["datasets_path"]
        train_datasets = training_parameters["train_datasets"]
        valid_datasets = training_parameters["valid_datasets"]
        oversampling = training_parameters["oversampling"]
        clear_models_dir = training_parameters["clear_models_dir"],
        validation_type = training_parameters["validation"]
        early_stopping = training_parameters["early_stopping"]
        cross_validation = validation_type == "cross-validation"

        labels_data = LabelsData(training_parameters["sec_label"], training_parameters["nonsec_label"])

        train_dataframe = read_data(
            datasets_path, train_datasets, labels_data,
            oversampling and not cross_validation)
        valid_dataframe = read_data(datasets_path, valid_datasets, labels_data)
        
        metrics_file_path = training_parameters["metrics_file"]

        configuration = TrainConfiguration(
            model_type=model_type,
            epochs=epochs,
            max_len=max_len,
            datasets_path=datasets_path,
            train_datasets=train_datasets,
            valid_datasets=valid_datasets,
            oversampling=oversampling,
            clear_models_dir=clear_models_dir,
            early_stopping=early_stopping,
            validation_type=validation_type,
            train_dataframe=train_dataframe,
            valid_dataframe=valid_dataframe,
            metrics_file_path=metrics_file_path,
            labels_data=labels_data,
        )

        return configuration