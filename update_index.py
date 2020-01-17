#!/usr/bin/env python
from constants import *

import os
import requests
import random
import pickle
import argparse

import pandas as pd
import json
import sklearn
import spacy

import numpy as np
import torch
import torch.nn.functional as F
from itertools import islice
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import *

spacy.prefer_gpu()

def load_paragraphs():
    with open(SQUAD_TRAIN) as f:
        doc = json.load(f)

    paragraphs = []
    for topic in doc["data"]:
        for pgraph in topic["paragraphs"]:
            paragraphs.append(pgraph["context"])
    return paragraphs

def lemmatize(phrase):
    return " ".join([word.lemma_ for word in phrase])

def cache_lemmas(force=False):
    if force or not os.path.isfile(LEMMA_CACHE):
        sp = spacy.load("en_core_web_sm")
        paragraphs = load_paragraphs()
        docs = sp.pipe(tqdm(paragraphs))
        lemmas = [lemmatize(par) for par in docs]
        df = pd.DataFrame(data={'context': paragraphs, 'lemmas': lemmas})
        df.to_feather(LEMMA_CACHE)
    else:
        df = pd.read_feather(LEMMA_CACHE)

    return df.lemmas

def tfidf_index(lemmas, force=False):
    if force or not os.path.isfile(VECTOR_CACHE):
        vectorizer = TfidfVectorizer(
            stop_words='english', min_df=5, max_df=.5, ngram_range=(1,3))
        tfidf = vectorizer.fit_transform(lemmas)
        with open(VECTOR_CACHE, "wb") as f:
            pickle.dump(dict(vectorizer=vectorizer, tfidf=tfidf), f)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default=False, action='store_true')
    args = ap.parse_args()

    lemmas = cache_lemmas(force=args.force)
    tfidf_index(lemmas, force=args.force)
