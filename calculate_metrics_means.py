import os
import pandas as pd
from tabulate import tabulate

columns = ['loss', 'acc', 'f1', 'precision', 'recall', 'runtime', 'samples_per_second', 'epoch']
resulting_columns = ["Model", "Invalid predictions", "Accuracy", "Precision", "Recall", "F1-score"]
overall_cross = pd.DataFrame(columns=resulting_columns)
overall_lopo = pd.DataFrame(columns=resulting_columns)
overall_pfold = pd.DataFrame(columns=resulting_columns)

def parse_model_name(name):
    model_type = 'T5-small' if 'small' in name else 'T5-base'
    if '_tf' in name:
        classes = '-tf'
    elif '_01' in name:
        classes = '-01'
    else:
        classes = ''
    oversampling = '-os' if '_os' in name else ''
    return f'{model_type}{classes}{oversampling}'

METRICS_FOLDER = "metrics"

for path in sorted(os.listdir(METRICS_FOLDER)):
    full_path = os.path.join(METRICS_FOLDER, path)
    if os.path.isdir(full_path):
        continue
    data = pd.read_csv(full_path, names=columns, header=None)
    data = data[data['f1'] != 0]
    if 'cross' in path:
        overall = overall_cross
    elif 'lopo' in path:
        overall = overall_lopo
    else:
        overall = overall_pfold
    name = parse_model_name(path)
    invalid = 'inverse' if 'reverse' in path else 'security'
    overall.loc[len(overall)] = [name, invalid, 100 * data["acc"].mean(), 100 * data["precision"].mean(), 100 * data["recall"].mean(), 100 * data["f1"].mean()]
    # print(full_path, "f1: ", data["f1"].mean(), " precision: ", data["precision"].mean(), " recall: ", data["recall"].mean())

# print("NoRBERT cross-evaluation f1: 91%, precision: 90%, recall: 92%")
# print("NoRBERT loPo-evaluation f1: 87%, precision: 82%, recall: 92%")
overall_cross = overall_cross.round(2)
overall_lopo = overall_lopo.round(2)
overall_pfold = overall_pfold.round(2)
overall_cross.loc[len(overall_cross)] = ["NoRBERT", "-", "-", 92, 90, 91]
overall_lopo.loc[len(overall_lopo)] = ["NoRBERT", "-", "-", 92, 82, 87]
overall_pfold.loc[len(overall_pfold)] = ["NoRBERT", "-", "-", 88, 92, 90]
print(tabulate(overall_cross, headers = 'keys', tablefmt = 'psql'))
print(tabulate(overall_lopo, headers = 'keys', tablefmt = 'psql'))
print(tabulate(overall_pfold, headers = 'keys', tablefmt = 'psql'))
overall_cross.to_latex('metrics/tables/cross.tex', index=False)
overall_lopo.to_latex('metrics/tables/lopo.tex', index=False)
overall_pfold.to_latex('metrics/tables/pfold.tex', index=False)