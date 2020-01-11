from unidecode import unidecode

def lemmatize(phrase):
    return " ".join([word.lemma_ for word in phrase])

def lemmatize_preproc(doc):
    return [unidecode(tok.lemma_.lower()) for tok in doc if not tok.is_stop]

