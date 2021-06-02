import os
import pandas as pd
from tabulate import tabulate

columns = ['loss', 'acc', 'f1', 'precision', 'recall', 'runtime', 'samples_per_second', 'epoch']

paths = ["cross_reverse_metrics.csv", "cross_security_metrics.csv", "p_reverse_metrics.csv", "p_security_metrics.csv"]
models = ['t5-small', 't5-base', 't5-base-true-false', 't5-base-true-false-os', 't5-base-tf-os-es']

overall = pd.DataFrame(columns=["name", "acc", "precision", "recall", "f1"])

for path in os.listdir("metrics"):
    full_path = os.path.join("metrics", path)
    if os.path.isdir(full_path):
        continue
    data = pd.read_csv(full_path, names=columns, header=None)
    data = data[data['f1'] != 0]
    overall.loc[len(overall)] = [path, 100 * data["acc"].mean(), 100 * data["precision"].mean(), 100 * data["recall"].mean(), 100 * data["f1"].mean()]
    # print(full_path, "f1: ", data["f1"].mean(), " precision: ", data["precision"].mean(), " recall: ", data["recall"].mean())

# print("NoRBERT cross-evaluation f1: 91%, precision: 90%, recall: 92%")
# print("NoRBERT loPo-evaluation f1: 87%, precision: 82%, recall: 92%")
overall.loc[len(overall)] = ["NoRBERT cross-evaluation", "-", 92, 90, 91]
overall.loc[len(overall)] = ["NoRBERT loPo-evaluation", "-", 92, 82, 87]
print(tabulate(overall, headers = 'keys', tablefmt = 'psql'))