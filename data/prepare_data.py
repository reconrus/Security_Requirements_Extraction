import os
import re
from functools import partial 
from typing import Dict
from xml.etree import ElementTree as ET

import arff
import argparse
import pandas as pd

SEC_LABEL = "sec"
NONSEC_LABEL = "nonsec"
COLUMNS = ["Text", "Label"]
ALL_KEY = "all"


def setup_parser(parser):
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
        "--owasp",
        default="./Datasets/OWASP",
        type=str,
        help="Path to OWASP Application Security Verification Standard folder",
    )
    parser.add_argument(
        "-o",
        default="processed",
        type=str,
        help="Output folder",
    )


def read_secreq(path) -> Dict[str, pd.DataFrame]:
    read_documents = {}
    secreq_dataset = pd.DataFrame(columns=COLUMNS)
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        dataset = pd.read_csv(
            filepath,
            sep=";",
            header=None,
            names=COLUMNS,
            engine="python",
        )
        dataset['Label'].replace('xyz', 'sec', inplace=True)
        dataset['Text'] = dataset['Text'].apply(str.strip)
        dataset = dataset.dropna()
        read_documents[filename] = dataset
        secreq_dataset = secreq_dataset.append(dataset)

    read_documents[ALL_KEY] = secreq_dataset
    return read_documents


def read_promise(path) -> Dict[str, pd.DataFrame]:
    data = arff.load(open(path, "r", encoding="cp1252"))
    adjust_class = lambda x: SEC_LABEL if x == "SE" else NONSEC_LABEL
    data = [[row[0], row[1].strip(), adjust_class(row[2])] for row in data["data"]]
    promise_dataset = pd.DataFrame(data, columns=['document'] + COLUMNS)
    
    documents = {ALL_KEY: promise_dataset[COLUMNS]}
    for i in range(1, 11):
        document = promise_dataset[promise_dataset['document'] == str(i)][COLUMNS]
        documents[str(i)] = document
    return documents


def parse_concord_xml(path) -> pd.DataFrame:
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
            is_requirement = False
            for feature in annotation:
                if (
                    feature.find("Value").text == "yes"
                    and feature.find("Name").text == "security"
                ):
                    is_sec = True
                    is_requirement = True
                    break
                if feature.find("Value").text == "yes":
                    is_requirement = True

            class_ = SEC_LABEL if is_sec else NONSEC_LABEL
            if is_requirement:
                data.append([units[start_node], class_])
    dataset = pd.DataFrame(data, columns=COLUMNS)
    return dataset


def read_concord(path) -> Dict[str, pd.DataFrame]:
    read_documents = {}
    concord_dataset = pd.DataFrame(columns=COLUMNS)
    for filename in os.listdir(path):
        if not filename.endswith("xml"):
            continue
        dataset = parse_concord_xml(os.path.join(path, filename))
        read_documents[filename] = dataset
        concord_dataset = concord_dataset.append(dataset)
    read_documents[ALL_KEY] = concord_dataset
    return read_documents


def read_cchit(path) -> Dict[str, pd.DataFrame]:
    columns = ["Criteria #", "Criteria", "Comments"]
    cchit_data = pd.read_excel(path, header=5, usecols=columns)
    cchit_data = cchit_data[cchit_data[columns[0]].notna()].dropna()

    prepare_label = lambda criteria: SEC_LABEL if "SC" in criteria else NONSEC_LABEL

    def prepare_text(texts):
        if type(texts[1]) == str:
            return f"{texts[0].strip()} {texts[1].strip()}".replace("\n", " ")
        else:
            return texts[0].strip().replace("\n", " ")

    labels = cchit_data[columns[0]].map(prepare_label)
    texts = cchit_data[columns[1:]].apply(prepare_text, axis=1)

    data = {COLUMNS[0]: texts, COLUMNS[1]: labels}
    cchit_dataset = pd.DataFrame(data).dropna()
    return {ALL_KEY: cchit_dataset}


def prepare_owasp_text(text: str) -> str:
    verify_pattern = "^(Verify that)|^(Verify)"
    link_pattern = "\(\[(C\d+(, )*)+].*\)$"
    text = re.sub(f'{verify_pattern}|{link_pattern}', "", text).strip()
    return text.title()


def read_owasp_v4(path, owasp_dataset):
    owasp_v4_data = pd.read_csv(path, sep=",", usecols=["req_description"])
    owasp_v4_data = owasp_v4_data.rename(columns={"req_description": "Text"})

    owasp_v4_data["Text"] = owasp_v4_data["Text"].apply(prepare_owasp_text)
    return owasp_dataset.append(owasp_v4_data)


def read_owasp_v3(path, owasp_dataset):
    columns = ["Detail"]
    owasp_v3_data = pd.read_excel(path, usecols=columns)
    owasp_v3_data = owasp_v3_data.rename(columns={"Detail": "Text"})
    owasp_v3_data.reset_index()
    columns_upd = ["Description"]
    stop_phrases = '|'.join(
        ["Business Logic Section", "Deprecated", "EMPTY REQUIREMENT"])
    owasp_v3_upd_data = pd.read_excel(path, sheet_name=1, usecols=columns_upd)
    owasp_v3_upd_data = owasp_v3_upd_data[~owasp_v3_upd_data["Description"].str.contains(
        stop_phrases)]
    owasp_v3_upd_data = owasp_v3_upd_data.rename(
        columns={"Description": "Text"})
    owasp_v3_upd_data.reset_index()

    owasp_v3_data = owasp_v3_data.append(owasp_v3_upd_data)
    owasp_v3_data["Text"] = owasp_v3_data["Text"].apply(prepare_owasp_text)
    return owasp_dataset.append(owasp_v3_data)


def read_owasp(path) -> Dict[str, pd.DataFrame]:
    owasp_dataset = pd.DataFrame(columns=["Text"])
    path_v3 = os.path.join(path, "OWASP_3.0.1.xls")
    path_v4 = os.path.join(path, "OWASP_4.0.csv")
    owasp_dataset = read_owasp_v4(path_v4, owasp_dataset)
    owasp_dataset = read_owasp_v3(path_v3, owasp_dataset)
    owasp_dataset = owasp_dataset.drop_duplicates()
    owasp_dataset["Label"] = SEC_LABEL
    return {ALL_KEY: owasp_dataset}


def write_documents(documents: Dict[str, pd.DataFrame], path: str):
    if not os.path.isdir(path):
        os.mkdir(path)
    for original_name, dataframe in documents.items():
        name = "{}.csv".format(os.path.splitext(original_name)[0])
        document_path = os.path.join(path, name)
        dataframe.to_csv(document_path, sep="\t", index=False)


def read_and_save_datasets(args):
    dataset_names = ["secreq", "promise", "concord", "cchit", "owasp"]
    read_functions = {
        "secreq": partial(read_secreq, args.sec_req),
        "promise": partial(read_promise, args.promise),
        "concord": partial(read_concord, args.concord),
        "cchit": partial(read_cchit, args.cchit),
        "owasp": partial(read_owasp, args.owasp),
    }

    merged_dataset = pd.DataFrame(columns=COLUMNS)

    if not os.path.isdir(args.o):
        os.mkdir(args.o)

    for dataset_name in dataset_names:
        documents = read_functions[dataset_name]()
        path = os.path.join(args.o, dataset_name)
        write_documents(documents, path)
        merged_dataset = merged_dataset.append(documents["all"])

    merged_dataset = merged_dataset.drop_duplicates()
    joined_path = os.path.join(args.o, "joined")
    if not os.path.isdir(joined_path):
        os.mkdir(joined_path)
    merged_dataset_path = os.path.join(joined_path, "all.csv")
    merged_dataset.to_csv(merged_dataset_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data unification script for Security Requirements Extraction task"
    )
    setup_parser(parser)
    args = parser.parse_args()
    read_and_save_datasets(args)
