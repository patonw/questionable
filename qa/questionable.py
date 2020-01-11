#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from transformers import *

from .constants import *
from .util import *
from .indexers import *
from .context_source import *

class AnsweringMachine:
    def __init__(self, docs):
        self.docs = docs
        print("Loading language models")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering \
                .from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') \
                .to(DEVICE)

    def assemble_contexts(self, question, contexts):
        question_df = pd.DataFrame.from_records([{'question': question, 'context': self.docs[ctx].text} for (ctx,s) in contexts])
        question_df["encoded"] = question_df.apply(lambda row: self.tokenizer.encode("[CLS] " + row["question"] + " [SEP] " + row["context"] + " [SEP]"), axis=1)
        question_df["tok_type"] = question_df.apply(lambda row: [0 if i <= row["encoded"].index(102) else 1 for i in range(len(row["encoded"]))], axis=1)
        return question_df

    def decode_answer(self, row):
        input_ids = row.encoded
        offset = row.answer_start
        length = np.clip(row.answer_length, 0, 20)
        return self.tokenizer.decode(input_ids[offset:][:length])

    def map_answers(self, question_df, prune=True, cutoff=1.0, debug=False):
        with torch.no_grad():
            X = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in question_df["encoded"]], batch_first=True).to(DEVICE)
            T = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in question_df["tok_type"]], batch_first=True).to(DEVICE)
            start_scores, end_scores = self.model(X, token_type_ids=T)
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
            answer_df["answer"] = answer_df.apply(self.decode_answer, axis=1)
        
        return answer_df
