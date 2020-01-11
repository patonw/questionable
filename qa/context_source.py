from spacy.tokens import DocBin

from .constants import *

class ContextSource:
    def __init__(self, sp, docs):
        self.sp = sp
        self.docs = docs

    def from_cache(sp):
        sp.vocab.from_disk(VOCAB_CACHE)

        with open(DOCBIN_CACHE, "rb") as f:
            bb = f.read()
            doc_bin = DocBin().from_bytes(bb)
        docs = list(doc_bin.get_docs(sp.vocab))

        return ContextSource(sp, docs)
