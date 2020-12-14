#!/bin/sh

mkdir RawDatasets
cd RawDatasets

# SecReq Dataset
if [ ! -d SecReq ]; then
    wget -O SecReq.zip https://www.dropbox.com/sh/mcvx5ium0zx7bly/AABfJaFt0nWvjiNJs1RUYf_Pa?dl=1
    unzip SecReq.zip -d SecReq
    rm SecReq.zip
fi

# # PROMISE Dataset
if [ ! -d nfr ]; then
    wget -O Promise.tar https://zenodo.org/record/268542/files/nfr.tar?download=1
    tar -xvf Promise.tar
    rm Promise.tar
    sed -i '45s/.*/@ATTRIBUTE class {F,A,L,LF,MN,O,PE,SC,SE,US,FT,PO}/' ./nfr/nfr.arff
fi

# Concordia Dataset
if [ ! -d NFRClassifier ]; then
    wget -O Concordia.tar.gz https://www.semanticsoftware.info/system/files/NFRClassifier.tar.gz
    tar -xzf Concordia.tar.gz
    rm Concordia.tar.gz
fi

# CCHIT Dataset
if [ ! -f CCHIT.xls ]; then
    wget -O CCHIT.xls https://www.dropbox.com/s/7pe4xq0ntwbbrlx/CCHIT%20Certified%202011%20Ambulatory%20EHR%20Criteria%2020110517.xls?dl=1
fi

# OWASP Application Security Verification Standard
if [ ! -d OWASP ]; then
    mkdir OWASP && cd OWASP
    wget -O OWASP_3.0.1.xlsx https://github.com/OWASP/ASVS/blob/master/3.0.1/ASVS-excel-v3.0.1.xlsx?raw=true
    wget -O OWASP_4.0.csv https://raw.githubusercontent.com/OWASP/ASVS/v4.0.2/4.0/docs_en/OWASP%20Application%20Security%20Verification%20Standard%204.0.2-en.csv
    cd ..
fi

cd ..
[ ! -d env ] && python3 -m venv env
. ./env/bin/activate
pip3 install pandas liac-arff xlrd
python3 prepare_data.py --sec_req ./RawDatasets/SecReq --promise ./RawDatasets/nfr/nfr.arff \
    --concord ./RawDatasets/NFRClassifier/gate/application-resources/Requirements/ \
    --cchit ./RawDatasets/CCHIT.xls --owasp ./RawDatasets/OWASP
