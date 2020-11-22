import os

import arff
import argparse
import pandas as pd

from xml.etree import ElementTree as ET

parser = argparse.ArgumentParser(
    description="Data unification script for Security Requirements Extraction task"
)
parser.add_argument(
    "--sec_req",
    default="./Datasets/SecReq",
    type=str,
    help="Path to folder with CPN, ePurse and GPS datasets",
)
parser.add_argument(
    "--promise",
    default="./Datasets/PROMISE/nfr/nfr.arff",
    type=str,
    help="Path to extracted PROMISE arff file. \nFor now there is a need to manually delete comment on line 45",
)
parser.add_argument(
    "--concord",
    default="./Datasets/NFRClassifier/gate/application-resources/Requirements/",
    type=str,
    help="Path to folder with Concord xml data files",
)
parser.add_argument(
    "--cchit",
    default="./Datasets/CCHIT.xls",
    type=str,
    help="Path to CCHIT Excel Sheet",
)
parser.add_argument(
    "-o",
    default="result.csv",
    type=str,
    help="Output file path",
)
parser.add_argument(
    "--min_len",
    default=3,
    type=int,
    help="Minimum number of characters in a classification unit",
)
args = parser.parse_args()


def read_seqreq(path, resulting_dataset):
    for f in os.listdir(path):
        filepath = os.path.join(path, f)
        dataset = pd.read_csv(
            filepath,
            sep=";",
            header=None,
            names=resulting_dataset.columns,
            engine="python",
        )
        resulting_dataset = resulting_dataset.append(dataset)
    return resulting_dataset.dropna()


def read_promise(path, resulting_dataset):
    data = arff.load(open(path, "r", encoding="cp1252"))
    adjust_class = lambda x: "sec" if x == "SE" else "nonsec"
    data = [[row[1].strip(), adjust_class(row[2])] for row in data["data"]]
    df = pd.DataFrame(data, columns=resulting_dataset.columns)
    return resulting_dataset.append(df)


def parse_concord_xml(path, resulting_dataset, min_len):
    tree = ET.parse(path)
    root = tree.getroot()
    units = dict()
    nodes = next(root.iter("TextWithNodes"))
    count = 0
    for child in nodes:
        if count % 2 == 0:
            units[child.attrib["id"]] = child.tail.strip()
        count += 1

    data = []
    for annotations in root.iter("AnnotationSet"):
        for annotation in annotations:
            start_node = annotation.get("StartNode")
            is_sec = False
            for feature in annotation:
                if (
                    feature.find("Value").text == "yes"
                    and feature.find("Name").text == "security"
                ):
                    is_sec = True
                    break
            class_ = "sec" if is_sec else "nonsec"
            if len(units[start_node]) > min_len:
                data.append([units[start_node], class_])
    df = pd.DataFrame(data, columns=resulting_dataset.columns)
    return resulting_dataset.append(df)


def read_concord(path, resulting_dataset, min_len):
    for filepath in os.listdir(path):
        if filepath.endswith("xml"):
            resulting_dataset = parse_concord_xml(
                os.path.join(path, filepath), resulting_dataset, min_len
            )
    return resulting_dataset


def read_cchit(path, resulting_dataset): 
    columns = ["Criteria #", "Criteria", "Comments"]
    cchit_data = pd.read_excel(path, header=5, usecols=columns)
    # cchit_data = pd.read_excel(path, header=5)
    cchit_data = cchit_data[cchit_data[columns[0]].notna()].dropna()    
    cchit_data.to_csv("wtf.csv")

    prepare_label = lambda x: "sec" if "SC" in x else "nonsec"
    prepare_text = lambda x: f"{x[0].strip()} {x[1].strip()}".replace("\n", " ") if type(x[1]) == str else x[0].strip().replace("\n", " ")
    labels = cchit_data[columns[0]].map(prepare_label)
    texts = cchit_data[columns[1:]].apply(prepare_text, axis=1)

    data = {resulting_dataset.columns[0]: texts, resulting_dataset.columns[1]: labels} 
    df = pd.DataFrame(data).dropna()
    return resulting_dataset.append(df)

def read_datasets(args):
    columns = ["Text", "Label"]
    resulting_dataset = pd.DataFrame(columns=columns)
    resulting_dataset = read_seqreq(args.sec_req, resulting_dataset)
    resulting_dataset = read_promise(args.promise, resulting_dataset)
    resulting_dataset = read_concord(args.concord, resulting_dataset, args.min_len)
    resulting_dataset = read_cchit(args.cchit, resulting_dataset)

    resulting_dataset = resulting_dataset.drop_duplicates()
    resulting_dataset.to_csv(args.o, sep="\t", index=False)


if __name__ == "__main__":
    read_datasets(args)