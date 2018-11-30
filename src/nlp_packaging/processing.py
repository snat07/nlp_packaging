import pandas as pd

from spacy.lang.en.stop_words import STOP_WORDS


def lemmatize(doc):
    return [token.lemma_ for token in doc 
            if not token.is_punct 
            and not token.is_space 
            and token.lower_ not in STOP_WORDS
            and not token.tag_ == "POS"]

def tf(string, doc):
    return lemmatize(doc).count(string.lower())

def idf(word, docs):
    word_in_documents = 0
    for doc in docs:
        if tf(word, doc) > 0:
            word_in_documents += 1
    if word_in_documents > 0:
        return 1/word_in_documents
    else:
        return 0

def tf_idf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)

def all_lemmas(docs):
    lemmas = []
    for doc in docs:
         lemmas.extend(lemmatize(doc))
    return set(lemmas)
  
def tf_idf_doc(doc, docs):
    all_lemmas_array = all_lemmas(docs)
    tf_idf_dict = {}
    for lemma in all_lemmas_array:
        tf_idf_dict[lemma] = tf_idf(lemma, doc, docs)
    return tf_idf_dict

def tf_idf_scores(docs):
    data = []
    for doc in docs:
        data.extend([tf_idf_doc(doc, docs)])
    return pd.DataFrame(data=data)