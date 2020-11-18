import os
import time
from utilities import poser, header, give_dependents, ider
from bs4 import BeautifulSoup
from collections import Counter

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0
distance_counter = Counter()
long_list = []
article_count = 0

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
                if token.has_attr('artificial') is False and token.has_attr('lemma'):
                    if token['lemma'] == 'ὁ' and poser(token) == 'article' and header(tokens, token) is not False:
                        article_count += 1
                        if poser(header(tokens, token)) == 'noun':
                            distance = int(ider(header(tokens, token))) - int(ider(token))
                            distance_counter[distance] += 1
                            if distance < -5 or distance > 9:
                                if header(tokens, token).has_attr('artificial') is False:
                                    long_list.append([file, sentence['id'], int(ider(token)), ider(header(tokens, token))])

print(article_count)
print(distance_counter)
print(long_list)
