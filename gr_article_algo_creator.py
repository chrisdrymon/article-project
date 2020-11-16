import os
from bs4 import BeautifulSoup
import time
from collections import Counter


pos0_dict = {'a': 'adj', 'n': 'noun', 'v': 'verb', 'd': 'adv', 'c': 'conj', 'g': 'conj', 'r': 'adposition', 'b': 'conj',
             'p': 'pronoun', 'l': 'article', 'i': 'interjection', 'x': 'other', 'm': 'numeral', 'e': 'interjection'}
pos1_dict = {'1': 'first', '2': 'second', '3': 'third'}
pos2_dict = {'s': 'singular', 'p': 'plural', 'd': 'dual'}
pos3_dict = {'p': 'present', 'i': 'imperfect', 'r': 'perfect', 'a': 'aorist', 'l': 'pluperfect', 'f': 'future', 't':
             'future perfect'}
pos4_dict = {'i': 'indicative', 's': 'subjunctive', 'n': 'infinitive', 'm': 'imperative', 'p': 'participle',
             'o': 'optative'}
pos5_dict = {'a': 'active', 'm': 'middle', 'p': 'passive', 'e': 'middle or passive'}
pos6_dict = {'m': 'masculine', 'f': 'feminine', 'n': 'neuter'}
pos7_dict = {'n': 'nominative', 'g': 'genitive', 'd': 'dative', 'v': 'vocative', 'a': 'accusative'}
proiel_pos_dict = {'A': 'adj', 'D': 'adv', 'S': 'article', 'M': 'numeral', 'N': 'noun', 'C': 'conj', 'G': 'conj',
                   'P': 'pronoun', 'I': 'interjection', 'R': 'adposition', 'V': 'verb'}

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0


def poser(f_word):
    """Returns the part-of-speech of a token."""
    f_pos = 'other'
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 0:
            pos0 = f_word['postag'][0]
            if pos0 in pos0_dict:
                f_pos = pos0_dict[pos0]
    elif f_word.has_attr('part-of-speech'):
        if len(f_word['part-of-speech']) > 0:
            pos0 = f_word['part-of-speech'][0]
            if pos0 in proiel_pos_dict:
                f_pos = proiel_pos_dict[pos0]
    return f_pos


def header(f_sentence, f_word):
    """Returns the id of a token's head."""
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


# The purpose is to extract data from the annotated corpora to be used to train a machine learning algorithm. Two goals
# are in focus. 1) Be able to correct identify the POS of any occurrence of the lemma ο. 2) Be able to identify the
# head of the lemma ο if it is acting as an article.
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
                if token.has_attr('artificial') is False:
                    if token['lemma'] == 'ὁ' and poser(token) == 'article' and header(tokens, token) is not False:
                        print(token['form'], header(tokens, token)['form'])
                        time.sleep(1)
