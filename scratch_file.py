import os
import time
from utilities import poser, header, caser, grammatical_number, mooder, verber
from bs4 import BeautifulSoup
from collections import Counter

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0
relation_counter = Counter()
token_count = 0
sub_verb_count = 0
mismatched_num_count = 0


for file in indir:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                token_count += 1
                if caser(token)[0] == 'nominative' and poser(header(tokens, token))[0] == 'verb':
                    if token['relation'] == 'SBJ' or token['relation'] == 'sub':
                        sub_verb_count += 1
                        if grammatical_number(token)[0] != grammatical_number(header(tokens, token))[0] and \
                                verber(header(tokens, token) != 'infinitive'):
                            mismatched_num_count += 1
                            print(token['form'], header(tokens, token)['form'], f'{mismatched_num_count} of '
                                                                                f'{sub_verb_count} mismatching.')
                            time.sleep(1)
