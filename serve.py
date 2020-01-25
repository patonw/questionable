#!/usr/bin/env python

import spacy
from flask import Flask, request
from transformers import *

from qa.embedders import *
from qa.indexers import *
from qa.questionable import *
from qa.context_source import *

sp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering \
        .from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') \
        .to(DEVICE)

contexts = ContextSource.from_cache(sp)
docs = contexts.docs
terms = TfIdfIndexer.from_cache(sp)
ents = EntityIndexer(terms.scorer, sp=sp, docs=docs, exclude=terms.get_vocab())
embedder = BertEmbedder(tokenizer, model)
embidx = EmbeddingIndexer(sp, embedder)
embidx.build_index(docs)

indexer = CompositeIndexer(sp, contexts.docs, dict(ENTITY=ents, TFIDF=terms, EMBED=embidx))
am = AnsweringMachine(contexts.docs, tokenizer, model)

app = Flask(__name__)

@app.route('/')
def display():
    return "Looks like it works!"

@app.route('/answer')
def answer():
    question = request.args.get("q")
    if not question:
        return "Query parameter 'q' is required\n", 400
    
    contexts = indexer.fetch_contexts(question)
        
    if len(contexts) < 1:
        return "No relevant information found\n", 404

    question_df = am.assemble_contexts(question, contexts)
    answer_df = am.map_answers(question_df).drop(columns=["encoded"])
    return answer_df.to_json(orient="records")

if __name__=='__main__':
    port = os.getenv('HTTP_PORT', 8765)
    host = os.getenv('HTTP_HOST', '0.0.0.0')
    app.run(host=host, port=port)
