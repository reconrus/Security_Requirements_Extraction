#!/bin/sh

mkdir RawDatasets
cd RawDatasets

# SecReq Dataset
wget -O SecReq.zip https://www.dropbox.com/sh/mcvx5ium0zx7bly/AABfJaFt0nWvjiNJs1RUYf_Pa?dl=1
unzip SecReq.zip -d SecReq
rm SecReq.zip

# # PROMISE Dataset
wget -O Promise.tar https://zenodo.org/record/268542/files/nfr.tar?download=1
tar -xvf Promise.tar
rm Promise.tar
sed -i '45s/.*/@ATTRIBUTE class {F,A,L,LF,MN,O,PE,SC,SE,US,FT,PO}/' ./nfr/nfr.arff

# Concordia Dataset
wget -O Concordia.tar.gz https://www.semanticsoftware.info/system/files/NFRClassifier.tar.gz
tar -xzf Concordia.tar.gz
rm Concordia.tar.gz

cd ..
python3 -m venv env
. ./env/bin/activate
pip3 install pandas liac-arff
python3 prepare_data.py --sec_req ./RawDatasets/SecReq --promise ./RawDatasets/nfr/nfr.arff --concord ./RawDatasets/NFRClassifier/gate/application-resources/Requirements/
