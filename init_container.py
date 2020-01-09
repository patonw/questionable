#!/usr/bin/python

import os
from constants import *

os.system("jupyter nbextension enable --py widgetsnbextension")
os.system("python -m spacy download en_core_web_sm")

if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isdir("cache"):
    os.mkdir("cache")
    
if not os.path.isfile(SQUAD_TRAIN):
    response = requests.get(SQUAD_URL, stream=True)

    with open(SQUAD_TRAIN, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)