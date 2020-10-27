import pandas as pd
import spacy
import os
from collections import Counter
import pickle
import json

df = pd.read_table(os.path.join('data', 'corpora', 'eng_wikipedia_2016_1M-sentences.txt'), '\t', index_col=0,
                   header=None, names=['Sentences'])
sentences = tuple(df['Sentences'].to_list())
nlp = spacy.load('en_core_web_sm')

abbreviation_list = []
abbreviation_dict = {}

# First checks if checkpoint save files exist in this directory. If so, this will pick up where those files
# last saved.

if {'sentence_counter.json', 'articular_counter.pickle', 'articular_lemma_counter.pickle', 'lemma_counter.pickle',
    'wordform_counter.pickle'}.issubset(os.listdir(os.getcwd())):
    with open("articular_counter.pickle", 'rb') as infile:
        articular_counter = pickle.load(infile)
    with open("articular_lemma_counter.pickle", 'rb') as infile:
        articular_lemma_counter = pickle.load(infile)
    with open("lemma_counter.pickle", 'rb') as infile:
        lemma_counter = pickle.load(infile)
    with open("wordform_counter.pickle", 'rb') as infile:
        wordform_counter = pickle.load(infile)
    with open("sentence_counter.json", 'r') as infile:
        sentence_counter = json.load(infile)
    sentences = sentences[sentence_counter:]
    print(f'Checkpoint files found. Starting data collection at sentence {sentence_counter}.')
else:
    wordform_counter = {}
    lemma_counter = {}
    articular_counter = {}
    articular_lemma_counter = {}
    sentence_counter = 0

# This creates a dictionary that counts its occurrences broken down by part of speech. It will look like this:
# {word1:{NN: 3, NNP: 7}, word2:{NN:1, NNP:2}}. It is only counting nouns and makes them all lower case.

for sentence in sentences:
    doc = nlp(sentence)
    for word in doc:
        l_lemma = word.lemma_.lower()
        if word.tag_ in ['NN', 'NNPS', 'NNP', 'NNS']:
            l_word = word.text.lower()
            if l_word in wordform_counter:
                wordform_counter[l_word][word.tag_] += 1
            else:
                wordform_counter[l_word] = Counter({word.tag_: 1})
            if l_lemma in lemma_counter:
                lemma_counter[l_lemma][word.tag_] += 1
            else:
                lemma_counter[l_lemma] = Counter({word.tag_: 1})
        elif l_lemma == 'the':
            l_head = word.head.text.lower()
            if word.head.tag_ in ['NN', 'NNPS', 'NNP', 'NNS']:
                l_head_lemma = word.head.lemma_.lower()
                if l_head in articular_counter:
                    articular_counter[l_head][word.head.tag_] += 1
                else:
                    articular_counter[l_head] = Counter({word.head.tag_: 1})
                if l_head_lemma in articular_lemma_counter:
                    articular_lemma_counter[l_head_lemma][word.head.tag_] += 1
                else:
                    articular_lemma_counter[l_head_lemma] = Counter({word.head.tag_: 1})

    # This gives some feedback to let the user known how far along the program has progressed.
    sentence_counter += 1
    if sentence_counter % 1000 == 0:
        print(f'Sentence {sentence_counter}!')

    # This may take hours to complete. This is here to create occasional backups in case it's interrupted.
    # These backups need to be saved as pickles because Counter objects get reloaded as normal dictionaries if they
    # are saved as JSONs. Usually, this is an easy fix, but since these are Counter objects nested within a dictionary,
    # it's quite a nuisance. Thus, the final files will be saved as both JSONs and pickles. I like JSON, but if the
    # files need edits in the future, preserving the Counters will be useful.
    if sentence_counter % 10000 == 0:
        with open("wordform_counter.pickle", "wb") as outfile:
            pickle.dump(wordform_counter, outfile)
        with open("lemma_counter.pickle", "wb") as outfile:
            pickle.dump(lemma_counter, outfile)
        with open("articular_counter.pickle", "wb") as outfile:
            pickle.dump(articular_counter, outfile)
        with open("articular_lemma_counter.pickle", "wb") as outfile:
            pickle.dump(articular_lemma_counter, outfile)
        with open("sentence_counter.json", 'w') as outfile:
            json.dump(sentence_counter, outfile)
        print('Backup created!')

# This will be a JSON of the grammatical abbreviations Spacy uses.
for lemma in lemma_counter:
    for key in lemma_counter[lemma]:
        if key not in abbreviation_list:
            abbreviation_list.append(key)

# This creates a dictionary of abbreviations that Spacy uses. It will be save as a JSON eventually.
for abbreviation in abbreviation_list:
    abbreviation_dict[abbreviation] = spacy.explain(abbreviation)

# Saves the files to the working directory. I'd recommend manually moving them to [cwd]/data/databases/english once they
# have completed.
with open('wordform_counter.json', 'w') as outfile:
    json.dump(wordform_counter, outfile)
with open('lemma_counter.json', 'w') as outfile:
    json.dump(lemma_counter, outfile)
with open('articular_counter.json', 'w') as outfile:
    json.dump(articular_counter, outfile)
with open('articular_lemma_counter.json', 'w') as outfile:
    json.dump(articular_lemma_counter, outfile)
with open('sentence_counter.json', 'w') as outfile:
    json.dump(sentence_counter, outfile)
with open('abbreviations.json', 'w') as outfile:
    json.dump(abbreviation_dict, outfile)

with open('wordform_counter.pickle', 'wb') as outfile:
    pickle.dump(wordform_counter, outfile)
with open('lemma_counter.pickle', 'wb') as outfile:
    pickle.dump(lemma_counter, outfile)
with open('articular_counter.pickle', 'wb') as outfile:
    pickle.dump(articular_counter, outfile)
with open('articular_lemma_counter.pickle', 'wb') as outfile:
    pickle.dump(articular_lemma_counter, outfile)
