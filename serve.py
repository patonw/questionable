#!/usr/bin/env python

import spacy
from flask import Flask, request

from qa.indexers import *
from qa.questionable import *
from qa.context_source import *

app = Flask(__name__)
sp = spacy.load('en_core_web_sm')
contexts = ContextSource.from_cache(sp)
indexer = CompositeIndexer(sp, contexts.docs)
am = AnsweringMachine(contexts.docs)

@app.route('/')
def display():
    return "Looks like it works!"

@app.route('/answer')
def answer():
    question = request.args.get("q")
    if not question:
        return "Query parameter 'q' is required\n", 400
    
    #contexts, query = indexer.fetch_contexts(question, debug=True)
    #if len(query) < 1 or query[0].size < 1:
    #    return "Ask a better question\n", 400
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
