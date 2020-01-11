#!/usr/bin/env python
import os
import argparse
import spacy

from qa.indexers import *

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default=False, action='store_true')
    args = ap.parse_args()

    spacy.prefer_gpu()
    sp = spacy.load("en_core_web_sm")
    builder = TfIdfIndexer(sp)

    docs = builder.cache_docbin(force=args.force)
    builder.build_cache(docs, force=args.force)
