import tensorflow as tf
import os
from bs4 import BeautifulSoup
from tensor_utils import header, poser, person, grammatical_number, tenser, mooder, voicer, gender, caser, lemmer
import numpy as np

# This program reviews the annotation of Ancient Greek treebanks. Specifically, it uses a neural network to find
# possible errors in identifying article heads.

# Enable this to run on CPU instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = os.path.join('data', 'tf_logs', 'm0.934val0.898')
model = tf.keras.models.load_model(model_path)
corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0
problems = []

# for file in indir:
#     if file[-4:] == '.xml':
#         work_samples = 0
#         article_count = 0
#         file_count += 1
#         print(file_count, file)
#         xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
with open(os.path.join('data', 'corpora', 'greek', 'annotated', 'appian-the_civil_wars-bc-1.0-1.4-bu1.xml'), 'r',
          encoding='utf-8') as xml_file:
    soup = BeautifulSoup(xml_file, 'xml')
sentences = soup.find_all('sentence')
for sentence in sentences:
    tokens = sentence.find_all(['word', 'token'])
    for token in tokens:
        # The search should ignore elliptical tokens
        if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
            # I am only interested in finding possible instances of the Greek article.
            if lemmer(token) == 'ὁ':
                # Record the morphological tags of the tokens around the ὁ as tensors to train an LSTM.
                window_sequence = []
                window_tokens = []
                head_tok = header(tokens, token)
                # Create the window around the ὁ
                article_index = tokens.index(token)
                window_start = article_index - 4
                window_end = article_index + 10
                # If part of the window exceeds the bounds of the sentence, return a blank tensor for that token
                while window_start < 0:
                    window_sequence.append([0] * 49)
                    window_tokens.append('OOS')
                    window_start += 1
                # We might not be getting the final token into the training data! Check this!
                while window_start <= window_end:
                    try:
                        token_tensor = poser(tokens[window_start])[1] + person(tokens[window_start])[1] + \
                                       grammatical_number(tokens[window_start])[1] + \
                                       tenser(tokens[window_start])[1] + mooder(tokens[window_start])[1] + \
                                       voicer(tokens[window_start])[1] + gender(tokens[window_start])[1] + \
                                       caser(tokens[window_start])[1]
                        window_sequence.append(token_tensor)
                        window_tokens.append(tokens[window_start])
                    except IndexError:
                        # print([0]*49, 'Out of Sentence')
                        window_sequence.append([0] * 49)
                        window_tokens.append('OOS')
                    window_start += 1
                predictions = model.predict([window_sequence])
                try:
                    window_tok_head = window_tokens[int(np.argmax(predictions[0]))]
                except IndexError:
                    window_tok_head = 'other'
                if window_tok_head != head_tok:
                    print(f'\nIncorrect Prediction for {token["form"]} in Sentence: {sentence["id"]}, '
                          f'Token: {token["id"]}:')
                    for i, tok in enumerate(window_tokens):
                        try:
                            print(f'{tok["form"]}: {predictions[0][i]:.02%}')
                        except TypeError:
                            print(f'Empty: {predictions[0][i]:.02%}')
                    print(f'Other: {predictions[0][-1]:.02%}')
                    problems.append([sentence['id'], token['id'], np.amax(predictions[0])])
print('\nSummary of Instances Worthy of Review:')
for item in problems:
    print(f'Sentence: {item[0]}, Token: {item[1]}, Max Confidence: {item[2]:.02%}')
print(f'{len(problems)} Instances!')
