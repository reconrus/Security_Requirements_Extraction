import os
import pandas as pd
from tabulate import tabulate

columns = ['loss', 'acc', 'f1', 'precision', 'recall', 'runtime', 'samples_per_second', 'epoch']
resulting_columns = ["name", "acc", "precision", "recall", "f1"]
overall_cross = pd.DataFrame(columns=resulting_columns)
overall_lopo = pd.DataFrame(columns=resulting_columns)
overall_pfold = pd.DataFrame(columns=resulting_columns)

for path in sorted(os.listdir("metrics")):
    full_path = os.path.join("metrics", path)
    if os.path.isdir(full_path):
        continue
    data = pd.read_csv(full_path, names=columns, header=None)
    data = data[data['f1'] != 0]
    name = path.replace('.csv', '').replace('_', ' ')
    if 'cross' in name:
        overall = overall_cross
    elif 'lopo' in name:
        overall = overall_lopo
    else:
        overall = overall_pfold   
    overall.loc[len(overall)] = [name, 100 * data["acc"].mean(), 100 * data["precision"].mean(), 100 * data["recall"].mean(), 100 * data["f1"].mean()]
    # print(full_path, "f1: ", data["f1"].mean(), " precision: ", data["precision"].mean(), " recall: ", data["recall"].mean())

# print("NoRBERT cross-evaluation f1: 91%, precision: 90%, recall: 92%")
# print("NoRBERT loPo-evaluation f1: 87%, precision: 82%, recall: 92%")
overall_cross.loc[len(overall)] = ["NoRBERT cross-evaluation", "-", 92, 90, 91]
overall_lopo.loc[len(overall)] = ["NoRBERT loPo-evaluation", "-", 92, 82, 87]
overall_pfold.loc[len(overall)] = ["NoRBERT pfold-evaluation", "-", 88, 92, 90]
print(tabulate(overall_cross, headers = 'keys', tablefmt = 'psql'))
print(tabulate(overall_lopo, headers = 'keys', tablefmt = 'psql'))
print(tabulate(overall_pfold, headers = 'keys', tablefmt = 'psql'))
