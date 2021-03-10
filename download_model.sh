if [ ! -d models ]; then
    mkdir models    
fi

cd models
wget -O t5-small.pt.zip https://www.dropbox.com/s/k1srgi4c861odeu/t5-small-full.pt.zip?dl=0
unzip t5-small-full.pt.zip
cd ..

python3 -m spacy download en_core_web_sm