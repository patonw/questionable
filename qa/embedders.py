import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchtext.data import Field, Dataset, BucketIterator, Example, RawField

from transformers import *

from qa.constants import *


class VocabSTOI:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __getitem__(self, key):
        return self.tok.convert_tokens_to_ids([key])[0]


class VocabITOS:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __getitem__(self, key):
        return self.tok.convert_ids_to_tokens([key])[0]


class BertVocab:
    def __init__(self, tokenizer):
        self.itos = VocabITOS(tokenizer)
        self.stoi = VocabSTOI(tokenizer)


def ident(x):
    return x


class BertTokenizationPreprocessing:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, s):
        return ["[CLS]"] + self.tok.tokenize(s)[:510] + ["[SEP]"]


class SpacyBertField(Field):
    """Transforms spaCy documents into Bert token lists"""
    def __init__(self, tokenizer, preprocessing=None, **kwargs):

        if not preprocessing:
            preprocessing = BertTokenizationPreprocessing(tokenizer)

        super().__init__(pad_token=tokenizer.pad_token,
                         preprocessing=preprocessing,
                         tokenize=ident,
                         batch_first=True,
                         **kwargs)
        self.vocab = BertVocab(tokenizer)

    def build_vocab(self, *args, **kw):
        pass


REDUCTION_DIMS = 1024


def reduce_embeds(toks, emb):
    N = (toks != 0).sum(axis=1, keepdim=True)
    sumq = emb.sum(axis=1)
    meanq = sumq / N
    maxq, _ = emb.max(axis=1)
    minq, _ = emb.min(axis=1)
    #return torch.cat([meanq, minq, maxq], axis=1)
    return maxq


class Examplifier:
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, args):
        (i, doc) = args
        return Example.fromlist([i, doc], self.fields)


class BertEmbedder:
    def __init__(self, tokenizer, model, device=DEVICE):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def __call__(self, docs, progress=True, parallel=True):
        texts = [
            ' '.join([tok.lemma_ for tok in doc if not tok.is_stop])
            for doc in docs
        ]
        fields = [('index', RawField()),
                  ('context', SpacyBertField(self.tokenizer))]

        if parallel:
            with mp.Pool() as pool:
                examples = pool.map(Examplifier(fields),
                                    enumerate(tqdm(texts)))
        else:
            f = Examplifier(fields)
            examples = [f((i, t)) for (i, t) in enumerate(tqdm(texts))]

        ds = Dataset(examples, fields)
        buckets = BucketIterator(dataset=ds,
                                 batch_size=24,
                                 device=self.device,
                                 shuffle=False,
                                 sort=True,
                                 sort_key=lambda ex: -len(ex.context))

        embeds = np.zeros((len(texts), REDUCTION_DIMS), dtype=np.float32)
        for b in tqdm(buckets):
            with torch.no_grad():
                output = self.model.bert.embeddings(b.context)
                embeds[b.index] = reduce_embeds(b.context, output).cpu()

        return embeds

    def embed_sentence(self, query):
        query = ' '.join([word.lemma_ for word in query if not word.is_stop])
        query_ids = self.tokenizer.encode("[CLS] " + query + " [SEP]",
                                          add_special_tokens=False,
                                          max_length=512)
        X = torch.tensor(query_ids, device=self.device).unsqueeze(0)
        with torch.no_grad():
            query_emb = self.model.bert.embeddings(X)
            result = reduce_embeds(X, query_emb).cpu().numpy()

        return result
