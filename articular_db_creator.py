import pandas as pd
import spacy
import os
from collections import Counter
import pickle
import json
import time

df = pd.read_table(os.path.join('data', 'corpora', 'test_sentences.txt'), '\t', index_col=0, header=None,
                   names=['Sentences'])
sentences = tuple(df['Sentences'].to_list())
nlp = spacy.load('en_core_web_sm')

articular_count = {}
articular_lemma_count = {}
sentence_counter = 0

for sentence in sentences:
    doc = nlp(sentence)
    print(sentence)
    for word in doc:
        if word.lemma_ == 'the':
            if word.head.text in articular_count:
                articular_count[word.head.text][word.head.tag_] += 1
            else:
                articular_count[word.head.text] = Counter({word.head.tag_: 1})
            if word.head.lemma_ in articular_lemma_count:
                articular_lemma_count[word.head.lemma_][word.head.tag_] += 1
            else:
                articular_lemma_count[word.head.lemma_] = Counter({word.head.tag_: 1})
    print(articular_count)
    print(articular_lemma_count)
    time.sleep(3)
