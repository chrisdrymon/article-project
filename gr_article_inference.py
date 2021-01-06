import tensorflow as tf
import os
from bs4 import BeautifulSoup
from tensor_utils import header, poser, person, grammatical_number, tenser, mooder, voicer, gender, caser, lemmer

# Enable this to run on CPU instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = os.path.join('data', 'tf_logs', 'm0.934val0.898')
model = tf.keras.models.load_model(model_path)

with open(os.path.join('data', 'corpora', 'greek', 'annotated', 'aesop-fables_1-53-tlg0096.tlg002.xml'), 'r',
          encoding='utf-8') as xml_file:
    soup = BeautifulSoup(xml_file, 'xml')
test_sentence = soup.find_all('sentence')[1]
print(f'Sentence ID: {test_sentence["id"]}')
tokens = test_sentence.find_all(['word', 'token'])
for token in tokens:
    # The search should ignore elliptical tokens
    if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
        # I am only interested in finding possible instances of the Greek article.
        if lemmer(token) == 'ὁ':
            # Record the morphological tags of the tokens around the ὁ as tensors to train an LSTM.
            window_sequence = []
            window_tokens = []
            # The head could be any of the tokens in the 15-token wide window or an "other" category
            head_tensor = [0] * 16
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
            print(f'\nPredictions for {token["form"]}:')
            for i, tok in enumerate(window_tokens):
                try:
                    print(f'{tok["form"]}: {predictions[0][i]:.02%}')
                except TypeError:
                    print(f'Empty: {predictions[0][i]:.02%}')
