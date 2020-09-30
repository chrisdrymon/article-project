import pandas as pd
import spacy
import os
from collections import Counter
import pickle
import json

df = pd.read_table(os.path.join('data/corpora', 'eng_wikipedia_2016_1M-sentences.txt'), '\t', index_col=0, header=None,
                   names=['Sentences'])
sentences = tuple(df['Sentences'].to_list())
nlp = spacy.load('en_core_web_sm')
wordform_count = {}
lemma_count = {}
articular_count = {}
articular_lemma_count = {}
abbreviation_list = []
abbreviation_dict = {}
sentence_counter = 0

# We are creating a dictionary that looks like this:
# wordform = {word1:{NN: 3, NNP: 7}, word2:{NN:1, NNP:2}}

for sentence in sentences:
    doc = nlp(sentence)
    for word in doc:
        l_word = word.text.lower()
        l_lemma = word.lemma_.lower()
        if l_word in wordform_count:
            wordform_count[l_word][word.tag_] += 1
        else:
            wordform_count[l_word] = Counter({word.tag_: 1})
        if l_lemma in lemma_count:
            lemma_count[l_lemma][word.tag_] += 1
        else:
            lemma_count[l_lemma] = Counter({word.tag_: 1})
        if l_lemma == 'the':
            l_head = word.head.text.lower()
            l_head_lemma = word.head.lemma_.lower()
            if l_head in articular_count:
                articular_count[l_head][word.head.tag_] += 1
            else:
                articular_count[l_head] = Counter({word.head.tag_: 1})
            if l_head_lemma in articular_lemma_count:
                articular_lemma_count[l_head_lemma][word.head.tag_] += 1
            else:
                articular_lemma_count[l_head_lemma] = Counter({word.head.tag_: 1})

    sentence_counter += 1
    if sentence_counter % 1000 == 0:
        print(f'Sentence {sentence_counter}!')

    # This will take a few hours to complete. This is here to create occassional backups in case it's interrupted.
    # These backups need to be saved as pickles because Counter objects get reloaded as normal dictionaries if they
    # are saved as JSONs. The final files will be saved as JSONs though because I like JSON.
    if sentence_counter % 10000 == 0:
        with open("wordform_count.pickle", "wb") as outfile:
            pickle.dump(wordform_count, outfile)
        with open("lemma_count.pickle", "wb") as outfile:
            pickle.dump(lemma_count, outfile)
        with open("articular_counter.pickle", "wb") as outfile:
            pickle.dump(articular_count, outfile)
        with open("articular_lemma_counter.pickle", "wb") as outfile:
            pickle.dump(articular_lemma_count, outfile)
        with open("sentence_count.json", 'w') as outfile:
            json.dump(sentence_counter, outfile)
        print('Backup created!')

for lemma in lemma_count:
    for key in lemma_count[lemma]:
        if key not in abbreviation_list:
            abbreviation_list.append(key)

for abbreviation in abbreviation_list:
    abbreviation_dict[abbreviation] = spacy.explain(abbreviation)

with open("wordform_count.json", "w") as outfile:
    json.dump(wordform_count, outfile)
with open("lemma_count.json", "w") as outfile:
    json.dump(lemma_count, outfile)
with open("articular_counter.json", "w") as outfile:
    json.dump(articular_count, outfile)
with open("articular_lemma_counter.json", "w") as outfile:
    json.dump(articular_lemma_count, outfile)
with open("sentence_count.json", 'w') as outfile:
    json.dump(sentence_counter, outfile)
with open("abbreviations.json", 'w') as outfile:
    json.dump(abbreviation_dict, outfile)
