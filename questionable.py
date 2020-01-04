#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pickle
import random
import json
import numpy as np
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn.functional as F
from transformers import *
import spacy
from tqdm import tqdm

from flask import Flask, render_template, request

# In[2]:
# Please run the notebook "Questionable - Part 2" to generate this cache
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_TRAIN = "data/train-v2.0.json"
LEMMA_CACHE = "cache/lemmas.feather"
VECTOR_CACHE = "cache/vectors.pickle"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_candidates = 10 if device.type == 'cuda' else 5

# In[3]:

print("Loading language models")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(device)

# In[4]:


sp = spacy.load('en_core_web_sm')


# In[5]:


def lemmatize(phrase):
    return " ".join([word.lemma_ for word in sp(phrase)])


# In[6]:

print("Loading source texts")
df = pd.read_feather(LEMMA_CACHE)


# In[7]:


paragraphs = df.context
lemmas = df.lemmas
N = df.shape[0]


# In[9]:


if not os.path.isfile(VECTOR_CACHE):
    print("Indexing source texts")
    vectorizer = TfidfVectorizer(
        stop_words='english', min_df=5, max_df=.5, ngram_range=(1,3))
    tfidf = vectorizer.fit_transform(lemmas)
    with open(VECTOR_CACHE, "wb") as f:
        pickle.dump(dict(vectorizer=vectorizer, tfidf=tfidf), f)
else:
    with open(VECTOR_CACHE, "rb") as f:
        cache = pickle.load(f)
        tfidf = cache["tfidf"]
        vectorizer = cache["vectorizer"]

# In[11]:


def fetch_contexts(question, THRESH = 0.01, debug=False):
    query = vectorizer.transform([lemmatize(question)])
    scores = (tfidf * query.T).toarray()
    results = (np.flip(np.argsort(scores, axis=0)))
    candidate_idxs = [(i, scores[i]) for i in results[0:n_candidates, 0]]
    contexts = [(paragraphs[i],s) for (i,s) in candidate_idxs if s > THRESH]
    
    if debug:
        return contexts, vectorizer.inverse_transform(query)
    return contexts


# In[12]:


def assemble_contexts(question, contexts):
    question_df = pd.DataFrame.from_records([{'question': question, 'context': ctx} for (ctx,s) in contexts])
    question_df["encoded"] = question_df.apply(lambda row: tokenizer.encode("[CLS] " + row["question"] + " [SEP] " + row["context"] + " [SEP]"), axis=1)
    question_df["tok_type"] = question_df.apply(lambda row: [0 if i <= row["encoded"].index(102) else 1 for i in range(len(row["encoded"]))], axis=1)
    return question_df


# In[13]:


def decode_answer(row):
    input_ids = row.encoded
    offset = row.answer_start
    length = np.clip(row.answer_length, 0, 20)
    return tokenizer.decode(input_ids[offset:][:length])


# In[14]:


def map_answers(question_df, prune=True, cutoff=1.0, debug=False):
    with torch.no_grad():
        X = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in question_df["encoded"]], batch_first=True).to(device)
        T = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in question_df["tok_type"]], batch_first=True).to(device)
        start_scores, end_scores = model(X, token_type_ids=T)
        max_score, max_start = torch.max(start_scores, axis=1)
        soft_max = F.softmax(max_score, dim=0)
        
    start_scores = start_scores.cpu().numpy()
    max_start = max_start.cpu().numpy()
    max_score = max_score.cpu().numpy()
    answer_df = question_df[["context", "encoded"]].copy()
    answer_df["answer_score"] = max_score
    answer_df["answer_softmax"] = soft_max.cpu().numpy()
    answer_df["answer_start"] = max_start
    max_len = np.zeros_like(max_start)
    for i in range(max_start.shape[0]):
        max_len[i] = torch.argmax(end_scores[i,max_start[i]:]) + 1

    answer_df["answer_length"] = max_len
    
    if prune:
        answer_df = answer_df[answer_df.answer_softmax > (1.0 / answer_df.shape[0])]
        
    answer_df = answer_df[answer_df.answer_start != 0]
    answer_df = answer_df[answer_df.answer_score > cutoff]
    answer_df = answer_df.sort_values(by="answer_score", ascending=False)
    if answer_df.shape[0] > 0:
        answer_df["answer"] = answer_df.apply(decode_answer, axis=1)
    
    return answer_df


app = Flask(__name__)

@app.route('/')
def display():
    return "Looks like it works!"

@app.route('/answer')
def answer():
    question = request.args.get("q")
    if not question:
        return "Query parameter 'q' is required\n", 400
    
    contexts, query = fetch_contexts(question, debug=True)
    if len(query) < 1 or query[0].size < 1:
        return "Ask a better question\n", 400
        
    if len(contexts) < 1:
        return "No relevant information found\n", 404

    question_df = assemble_contexts(question, contexts)
    answer_df = map_answers(question_df).drop(columns=["encoded"])
    return answer_df.to_json(orient="records")

if __name__=='__main__':
    port = os.getenv('HTTP_PORT', 8765)
    app.run(port=port)