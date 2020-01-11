#!/usr/bin/python

import os
import multiprocessing as mp
import requests
from tqdm import tqdm
from transformers import *

from qa.constants import *

os.system("jupyter nbextension enable --py widgetsnbextension")
os.system("python3 -m spacy download en_core_web_sm")

if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)
    
def preload_bert():
    print("Downloading pretrained models to cache")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def download_squad():
    if not os.path.isfile(SQUAD_TRAIN):
        print(f"Downloading squad dataset as {SQUAD_TRAIN}")
        response = requests.get(SQUAD_URL, stream=True)

        with open(SQUAD_TRAIN, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

with mp.Pool() as pool:
    fut = pool.apply_async(preload_bert)
    download_squad()
    fut.wait()
