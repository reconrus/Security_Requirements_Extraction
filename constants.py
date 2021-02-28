import os

TRAINING_APPLICATION_NAME = "training-script"
PREDICTING_APPLICATION_NAME = "predicting-script"

MODEL_FOLDER = "model"

TMP_FOLDER_NAME = "tmp"
TRAIN_DATASET_PATH = os.path.join(TMP_FOLDER_NAME, "train_data.pt")
VALID_DATASET_PATH = os.path.join(TMP_FOLDER_NAME, "valid_data.pt")
PREDICT_DATASET_PATH = os.path.join(TMP_FOLDER_NAME, "predict_data.pt")

DEFAULT_EPOCHS = 20
MAX_LENGTH = 100
MODEL_TYPE = "t5-small"
MODEL_FILENAME = f"{MODEL_TYPE}.pt"

SEC_LABEL = "sec"
NONSEC_LABEL = "nonsec"
OTHER_LABEL = "other"
