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
    if file[-4:] == '.xml' and file[:3] == 'Nic':
        article_count = 0
        file_count += 1
        print(file_count, file)
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                token_count += 1
                if token.has_attr('artificial') is False:
                    if token['lemma'] == 'ὁ' and poser(token) == 'article':
                        article_count += 1
                        print(article_count)
