import pickle
import json

import nmslib
from tqdm import tqdm
import numpy as np
from spacy.tokens import DocBin
from spacy.strings import hash_string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .constants import *
from .util import *


class TfIdfIndexer:
    def __init__(self, sp, topk=TOPK):
        self.sp = sp
        self.vectorizer = None
        self.vectors = None
        self.topk = TOPK

    def load_cache(self):
        with open(VECTOR_CACHE, "rb") as f:
            cache = pickle.load(f)
            self.vectorizer = cache["vectorizer"]
            self.vectors = cache["tfidf"]

    def from_cache(sp):
        self = TfIdfIndexer(sp)
        self.load_cache()
        return self

    def fetch_contexts(self, question, THRESH=0.01, debug=False):
        tokens = self.sp(question)
        query = self.vectorizer.transform([tokens])
        scores = (self.vectors * query.T).toarray()
        results = (np.flip(np.argsort(scores, axis=0)))
        candidate_idxs = [(i, scores[i]) for i in results[0:self.topk, 0]]
        contexts = [(i, s.item()) for (i, s) in candidate_idxs if s > THRESH]

        if debug:
            return contexts, self.vectorizer.inverse_transform(query)
        return contexts

    def scorer(self, contexts, query):
        scores = self.vectorizer.transform(
            contexts) * self.vectorizer.transform([query]).transpose()
        scores = (np.asarray(scores.todense()).flatten())
        return scores

    def get_vocab(self):
        return self.vectorizer.vocabulary_

    def load_paragraphs(self):
        with open(SQUAD_TRAIN) as f:
            doc = json.load(f)

        paragraphs = []
        for topic in doc["data"]:
            for pgraph in topic["paragraphs"]:
                paragraphs.append(pgraph["context"])
        return paragraphs

    def cache_lemmas(self, force=False):
        sp = self.sp
        if force or not os.path.isfile(LEMMA_CACHE):
            paragraphs = self.load_paragraphs()
            docs = sp.pipe(tqdm(paragraphs))
            lemmas = [lemmatize(par) for par in docs]
            df = pd.DataFrame(data={'context': paragraphs, 'lemmas': lemmas})
            df.to_feather(LEMMA_CACHE)
        else:
            df = pd.read_feather(LEMMA_CACHE)

        return df.lemmas

    def cache_docbin(self, force=False):
        sp = self.sp
        refresh = force or not os.path.isfile(DOCBIN_CACHE) \
                        or not os.path.isdir(VOCAB_CACHE)

        if refresh:
            paragraphs = self.load_paragraphs()
            doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"],
                             store_user_data=True)
            for doc in sp.pipe(tqdm(paragraphs)):
                doc_bin.add(doc)
            with open(DOCBIN_CACHE, "wb") as f:
                f.write(doc_bin.to_bytes())
            sp.vocab.to_disk(VOCAB_CACHE)

        sp.vocab.from_disk(VOCAB_CACHE)

        with open(DOCBIN_CACHE, "rb") as f:
            bb = f.read()
            doc_bin = DocBin().from_bytes(bb)
        return list(doc_bin.get_docs(sp.vocab))

    def build_cache(self, docs, force=False):
        if os.path.isfile(VECTOR_CACHE) and not force:
            return

        vectorizer = TfidfVectorizer(analyzer=lemmatize_preproc,
                                     stop_words='english',
                                     min_df=10,
                                     max_df=.5,
                                     ngram_range=(1, 3))
        tfidf = vectorizer.fit_transform(docs)
        with open(VECTOR_CACHE, "wb") as f:
            pickle.dump(dict(vectorizer=vectorizer, tfidf=tfidf), f)


class EntityIndexer:
    N = 10_001
    NUMERICS = set([
        "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
    ])

    def __init__(self, scorer, sp, docs, topk=TOPK, exclude=[]):
        self.scorer = scorer
        self.sp = sp
        self.docs = docs
        self.topk = topk
        self.exclude = exclude

        self.transformer = CountVectorizer(analyzer=self.doc_entities,
                                           stop_words='english',
                                           max_df=10)
        self.vectors = self.transformer.fit_transform(docs)

        self.bucketize()

    def doc_entities(self, doc):
        ents = [e for e in doc.ents if e.label_ not in EntityIndexer.NUMERICS]
        result = (unidecode(w.lemma_.lower()) for s in ents for w in s
                  if w.is_alpha and not w.is_stop)
        return [w for w in result if w not in self.exclude]

    # Maybe cache?
    def bucketize(self):
        self.table = [set() for i in range(self.N)]
        for (i, words) in enumerate(
                tqdm(self.transformer.inverse_transform(self.vectors))):
            for w in words:
                h = hash_string(str(w))
                self.table[h % self.N].add(i)

    def contexts_by_entities(self, doc):
        """Returns a set of document ids that *might* be related to named entities in the pre-processed question"""
        ents = self.doc_entities(doc)
        buckets = [hash_string(word) % self.N for word in ents]
        return set([doc_id for slot in buckets for doc_id in self.table[slot]])

    def ranked_contexts_by_entities(self, query, thresh=1e-5):
        doc_ids = list(self.contexts_by_entities(query))
        if len(doc_ids) < 1:
            return []

        contexts = [self.docs[i] for i in doc_ids]
        scores = self.scorer(contexts, query)

        sort_scores = np.asarray(np.flip(
            (scores).argsort()))  # indices ranked by score
        useful = sort_scores[scores[sort_scores] >=
                             thresh]  # Filter out irrelevant scores
        top_indices = useful[:self.topk]
        return [(doc_ids[i], scores[i].item()) for i in top_indices]

    def fetch_contexts(self, question, THRESH=0.01, debug=False):
        doc = self.sp(question)
        query = self.doc_entities(doc)
        contexts = self.ranked_contexts_by_entities(doc, thresh=THRESH)

        if debug:
            return contexts, query
        return contexts


class TrigramRanker:
    def __init__(self, topk=20):
        self.topk = topk
        self.vectorizer = TfidfVectorizer(analyzer='char_wb',
                                          ngram_range=(3, 3))

    def __call__(self, contexts, question):
        chargram = self.vectorizer
        ctx_vec = chargram.fit_transform(contexts)
        query_vec = chargram.transform([question])
        scored = (ctx_vec * query_vec.transpose()).todense()
        ranks = np.asarray(scored).squeeze().argsort().astype(np.int)
        return np.flip(ranks)[:self.topk]


class CompositeIndexer:
    def __init__(self, sp, docs, children, topk=None):
        self.sp = sp
        self.docs = docs
        self.children = children
        self.topk = topk

        self.ranker = TrigramRanker(topk)

    # TODO avoid reparse
    def fetch_contexts(self, question, THRESH=0.01, debug=False):
        merge_ids = {
            c: tag
            for (tag, child) in self.children.items()
            for (c, s) in child.fetch_contexts(question, THRESH)
        }
        flat = list(merge_ids.items())
        contexts = [self.docs[c].text for (c, s) in flat]
        picks = self.ranker(contexts, question)
        print(f"Picking {self.topk} contexts: {picks}")
        return [flat[i] for i in picks]


class EmbeddingIndexer:
    def __init__(self, sp, embedder, topk=TOPK):
        self.sp = sp
        self.embedder = embedder
        self.topk = topk

    def build_index(self, docs):
        self.embeds = self.embedder(docs, parallel=True)

        bfidx = nmslib.init(method='brute_force', space='cosinesimil')
        bfidx.addDataPointBatch(self.embeds)
        bfidx.createIndex(print_progress=True)
        self.index = bfidx

    def fetch_contexts(self, question, THRESH=0.01):
        if isinstance(question, str):
            question = self.sp(question)
        query = self.embedder.embed_sentence(question)
        ids, dist = self.index.knnQuery(query, self.topk)

        return list(zip(ids, dist))
