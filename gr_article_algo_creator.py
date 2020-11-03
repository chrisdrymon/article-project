import os
from bs4 import BeautifulSoup
import time
from collections import Counter


pos0_dict = {'a': 'adj', 'n': 'noun', 'v': 'verb', 'd': 'adv', 'c': 'conj', 'g': 'conj', 'r': 'adposition', 'b': 'conj',
             'p': 'pronoun', 'l': 'article', 'i': 'interjection', 'x': 'other', 'm': 'numeral', 'e': 'interjection'}
pos2_dict = {'s': 'singular', 'p': 'plural', 'd': 'dual'}
pos4_dict = {'i': 'indicative', 's': 'subjunctive', 'n': 'infinitive', 'm': 'imperative', 'p': 'participle',
             'o': 'optative'}
proiel_pos_dict = {'A': 'adj', 'D': 'adv', 'S': 'article', 'M': 'numeral', 'N': 'noun', 'C': 'conj', 'G': 'conj',
                   'P': 'pronoun', 'I': 'interjection', 'R': 'adposition', 'V': 'verb'}

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0


# This returns the part-of-speech or the mood if the part-of-speech is a verb for a given word.
def poser(f_word):
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 0:
            pos0 = f_word['postag'][0]
            if pos0 in pos0_dict:
                f_pos = pos0_dict[pos0]
            else:
                f_pos = 'other'
        else:
            f_pos = 'other'
    elif f_word.has_attr('part-of-speech'):
        if len(f_word['part-of-speech']) > 0:
            pos0 = f_word['part-of-speech'][0]
            if pos0 in proiel_pos_dict:
                f_pos = proiel_pos_dict[pos0]
            else:
                f_pos = 'other'
        else:
            f_pos = 'other'
    else:
        f_pos = 'other'
    return f_pos


def header(f_sentence, f_word):
    return_head = False
    f_head_id = 0
    if f_word.has_attr('head'):
        f_head_id = f_word['head']
    if f_word.has_attr('head-id'):
        f_head_id = f_word['head-id']
    for f_head in f_sentence:
        if f_head.has_attr('id'):
            if f_head['id'] == f_head_id:
                return_head = f_head
    return return_head


def toker

for file in indir:
    if file[-4:] == '.xml' and file[:3] == 'Nic':
        file_count += 1
        print(file_count, file)
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                if token['lemma'] == 'ὁ' and poser(token) == 'article' and token.has_attr('artificial') is False and header(tokens, token) is not False:
                    print(token['form'], header(tokens, token)['form'])
                    time.sleep(1)
