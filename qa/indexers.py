import pickle
import json

from tqdm import tqdm
import numpy as np
from spacy.tokens import DocBin
from spacy.strings import hash_string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .constants import *
from .util import *

class TfIdfIndexer:
    def __init__(self, sp):
        self.sp = sp
        self.vectorizer = None
        self.vectors = None

    def load_cache(self):
        with open(VECTOR_CACHE, "rb") as f:
            cache = pickle.load(f)
            self.vectorizer = cache["vectorizer"]
            self.vectors = cache["tfidf"]

    def from_cache(sp):
        self = TfIdfIndexer(sp)
        self.load_cache()
        return self

    def fetch_contexts(self, question, THRESH = 0.01, topk=TOPK, debug=False):
        tokens = self.sp(question)
        query = self.vectorizer.transform([tokens])
        scores = (self.vectors * query.T).toarray()
        results = (np.flip(np.argsort(scores, axis=0)))
        candidate_idxs = [(i, scores[i]) for i in results[0:topk, 0]]
        contexts = [(i,s.item()) for (i,s) in candidate_idxs if s > THRESH]
        
        if debug:
            return contexts, self.vectorizer.inverse_transform(query)
        return contexts

    def scorer(self, contexts, query):
        scores = self.vectorizer.transform(contexts) * self.vectorizer.transform([query]).transpose()
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
            doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
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

        vectorizer = TfidfVectorizer(
            analyzer=lemmatize_preproc,
            stop_words='english', min_df=10, max_df=.5, ngram_range=(1,3))
        tfidf = vectorizer.fit_transform(docs)
        with open(VECTOR_CACHE, "wb") as f:
            pickle.dump(dict(vectorizer=vectorizer, tfidf=tfidf), f)

class EntityIndexer:
    N = 10_001
    NUMERICS = set(["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"])

    def __init__(self, scorer, sp, docs, exclude=[]):
        self.scorer = scorer
        self.sp = sp
        self.docs = docs
        self.exclude = exclude

        self.transformer = CountVectorizer(
            analyzer=self.doc_entities,
            stop_words='english', max_df=10)
        self.vectors = self.transformer.fit_transform(docs)

        self.bucketize()

    def doc_entities(self, doc):
        ents = [e for e in doc.ents if e.label_ not in EntityIndexer.NUMERICS]
        result = (unidecode(w.lemma_.lower()) for s in ents for w in s if w.is_alpha and not w.is_stop)
        return [w for w in result if w not in self.exclude]

    # Maybe cache?
    def bucketize(self):
        self.table = [set() for i in range(self.N)]
        for (i, words) in enumerate(tqdm(self.transformer.inverse_transform(self.vectors))):
            for w in words:
                h = hash_string(str(w))
                self.table[h % self.N].add(i)

    def contexts_by_entities(self, doc):
        """Returns a set of document ids that *might* be related to named entities in the pre-processed question"""
        ents = self.doc_entities(doc)
        buckets = [hash_string(word) % self.N for word in ents]
        return set([doc_id for slot in buckets for doc_id in self.table[slot]])

    def ranked_contexts_by_entities(self, query, topk=10, thresh=1e-5):
        doc_ids = list(self.contexts_by_entities(query))
        if len(doc_ids) < 1:
            return []

        contexts = [self.docs[i] for i in doc_ids]
        scores = self.scorer(contexts, query)

        sort_scores = np.asarray(np.flip((scores).argsort())) # indices ranked by score
        useful = sort_scores[scores[sort_scores] >= thresh] # Filter out irrelevant scores
        top_indices = useful[:topk]
        return [(doc_ids[i], scores[i].item()) for i in top_indices]

    def fetch_contexts(self, question, THRESH = 0.01, debug=False):
        doc = self.sp(question)
        query = self.doc_entities(doc)
        contexts = self.ranked_contexts_by_entities(doc, thresh=THRESH)

        if debug:
            return contexts, query
        return contexts

class TrigramScorer:
    def __call__(self, contexts, query):
        pass

class CompositeIndexer:
    def __init__(self, sp, docs):
        self.terms = TfIdfIndexer.from_cache(sp)
        self.ents = EntityIndexer(self.terms.scorer, sp=sp, docs=docs, exclude=self.terms.get_vocab())

    # TODO avoid reparse
    # TODO filter by topk
    def fetch_contexts(self, question, THRESH = 0.01, debug=False):
        ids_by_ent = self.ents.fetch_contexts(question, THRESH)
        ids_by_term = self.terms.fetch_contexts(question, THRESH)
        merge_ids = { c:s for (c,s) in (ids_by_term + ids_by_ent) }
        return merge_ids.items()

