from dataclasses import dataclass
from typing import Dict, List

import yaml

from dataset import read_data, LabelsData


class Datasets:
    def __init__(self, config_path: str, datasets_path: str, labels_data: LabelsData, oversample: bool):
        with open(config_path, "r") as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            parameters = yaml.load(file, Loader=yaml.FullLoader)
    
        self.train_datasets = parameters["train_datasets"]
        self.valid_datasets = parameters["valid_datasets"]

        print(f'Using train datasets: {self.train_datasets}')
        print(f'Using valid datasets: {self.valid_datasets}')

        self.train_dataframe = read_data(datasets_path, self.train_datasets, labels_data, oversample)
        self.valid_dataframe = read_data(datasets_path, self.valid_datasets, labels_data)


@dataclass
class TrainConfiguration:
    model_type: str
    epochs: int
    max_len: int
    datasets_path: str
    oversampling: bool
    clear_models_dir: bool
    early_stopping: bool
    seed: int
    validation_type: str
    metrics_folder: str
    metrics_file_path: str
    labels_data: LabelsData
    datasets: Datasets

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
        oversampling = training_parameters["oversampling"]
        clear_models_dir = training_parameters["clear_models_dir"],
        validation_type = training_parameters["validation"]
        early_stopping = training_parameters["early_stopping"]
        seed = training_parameters["seed"]
        cross_validation = validation_type == "cross-validation"

        labels_data = LabelsData(training_parameters["sec_label"], training_parameters["nonsec_label"])

        metrics_folder = training_parameters["metrics_folder"]
        metrics_file_path = training_parameters["metrics_file"]

        datasets = Datasets(config_path=training_parameters["datasets_config_path"],
                            datasets_path=datasets_path,
                            labels_data=labels_data,
                            oversample=oversampling and not cross_validation)

        configuration = TrainConfiguration(
            model_type=model_type,
            epochs=epochs,
            max_len=max_len,
            datasets_path=datasets_path,
            oversampling=oversampling,
            clear_models_dir=clear_models_dir,
            early_stopping=early_stopping,
            seed=seed,
            validation_type=validation_type,
            metrics_folder=metrics_folder,
            metrics_file_path=metrics_file_path,
            labels_data=labels_data,
            datasets=datasets,
        )

        return configuration
