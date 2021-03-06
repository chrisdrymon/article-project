import os
from bs4 import BeautifulSoup
import time
import pickle
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter
from tensor_utils import header, poser, person, grammatical_number, tenser, mooder, voicer, gender, caser, lemmer

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


# Create a custom model saver
class ModelSaver(tf.keras.callbacks.Callback):
    # A custom tensorflow model saver that returns useful information
    best_loss = 100
    best_acc = 0
    best_val_acc = 0
    best_epoch = 0
    best_lr = 0
    best_dropout = 0
    new_best = False

    def on_train_begin(self, logs=None):
        self.best_loss = 100
        self.best_acc = 0
        self.best_val_acc = 0
        self.new_best = False

    def on_epoch_end(self, epoch, logs=None):
        # Save the best model based on validation accuracy.
        if logs['val_accuracy'] > self.best_val_acc:
            self.best_val_acc = logs['val_accuracy']
            # os.path.join('data', 'tf_logs', "m{0:.3f}val{1:.3f}".format(logs['accuracy'], logs['val_accuracy']))
            model_name = os.path.join('data', 'models', "m{0:.3f}val{1:.3f}".format(logs['accuracy'],
                                                                                    logs['val_accuracy']))
            tf.keras.models.save_model(model, model_name, save_format='h5')
            # The following is the save command that doesn't work.
            # tf.keras.models.save_model(model, model_name)
            self.best_epoch = epoch + 1
            self.new_best = True
            print('\n\nModel saved at epoch', epoch + 1, 'with', self.best_val_acc, 'validation accuracy.\n')

    def on_train_end(self, logs=None):
        if self.new_best:
            print('\nBest Model saved at epoch', self.best_epoch, 'with', self.best_val_acc, 'validation accuracy.')


pos0_dict = {'l': 'article', 'n': 'noun', 'a': 'adj', 'r': 'adposition', 'c': 'conj', 'i': 'interjection',
             'p': 'pronoun', 'v': 'verb', 'd': 'adv', 'm': 'numeral', 'g': 'conj', 'b': 'conj',
             'x': 'other', 'e': 'interjection'}
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

# The purpose is to extract data from the annotated corpora to be used to train a machine learning algorithm. Two goals
# are in focus. 1) Be able to correctly identify the POS of any occurrence of the lemma ο. 2) Be able to identify the
# head of the lemma ο if it is acting as an article.

# 1.1) Consider all tokens which occur 4 words before the article to 10 words after the article [-4, 10]. Out of the
# 79,335 articles which have non-elliptical heads in this corpus, only 73 have heads which which occur outside of that
# window.
# 2.1) If the head is elliptical, recognize that.

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0
samples = []
labels = []
total_samples_count = 0
replace_works = []
pos_counter = Counter()
head_loc_counter = Counter()
# Search through every work in the annotated Greek folder
for file in indir:
    if file[-4:] == '.xml':
        work_samples = 0
        article_count = 0
        file_count += 1
        print(file_count, file)
        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                # The search should ignore elliptical tokens
                if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
                    # I am only interested in finding possible instances of the Greek article.
                    if lemmer(token) == 'ὁ':
                        pos_counter[poser(token)[0]] += 1
                        if poser(token)[0] == 'article':
                            article_count += 1
                        total_samples_count += 1
                        work_samples += 1
                        # Record the morphological tags of the tokens around the ὁ to train an LSTM.
                        window_sequence = []
                        # The head could be any of the tokens in the 15-token wide window or an "other" category
                        head_tensor = [0] * 16
                        # Create the window around the ὁ
                        article_index = tokens.index(token)
                        window_start = article_index - 4
                        window_end = article_index + 10
                        # If part of the window exceeds the bounds of the sentence, return a blank tensor for that token
                        while window_start < 0:
                            window_sequence.append([0] * 49)
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
                            except IndexError:
                                # print([0]*49, 'Out of Sentence')
                                window_sequence.append([0]*49)
                            window_start += 1
                        samples.append(window_sequence)

                        # Makes sure the head word is tagged. If not, return an "other" label.
                        try:
                            head = header(tokens, token)
                            # Checks if head word is explicit. If not, return an "other" label.
                            if head.has_attr('artificial') or head.has_attr('empty-token-sort'):
                                head_tensor[15] = 1
                                head_loc_counter['other'] += 1
                            else:
                                head_index = tokens.index(head)
                                relative_head_location = head_index - article_index
                                # Check if head is inside the window. If not, return an "other" label.
                                if -4 <= relative_head_location <= 10:
                                    head_tensor[relative_head_location + 4] = 1
                                    head_loc_counter[relative_head_location] += 1
                                else:
                                    head_tensor[15] = 1
                                    head_loc_counter['other'] += 1
                        except AttributeError:
                            head_tensor[15] = 1
                            head_loc_counter['other'] += 1
                        # Record the part-of-speech and the head of the ὁ as training labels for a multi-class LSTM
                        labels.append(head_tensor)
        print(f'Work Samples/Total Samples: {work_samples}/{total_samples_count}')
        print(f'Percent of Samples in Work are Articles: {(article_count/work_samples):.02%}')
print(f'Part-of-Speech of instances of ὁ: {pos_counter}')
print('Head Locations Relative to ὁ:')
print(head_loc_counter)

# Split data into an 80%/20% training/validation split.
split = int(.8*len(labels))
train_data = samples[:split]
val_data = samples[split:]
train_labels = labels[:split]
val_labels = labels[split:]

# Enter the samples and labels into Tensorflow to train a neural network
model = tf.keras.Sequential()
# Input_shape = (n_steps, n_features). Since I combined everything into one tensor, try 49 features for now.
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation='relu', dropout=.3),
                               input_shape=(15, 49)))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation='relu', dropout=.3)))
model.add(layers.Bidirectional(layers.LSTM(128, activation='relu')))
model.add(layers.Dense(16, activation='softmax'))
modelSaver = ModelSaver()

log_dir = "data\\tf_logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
# log_dir = os.path.join('data', 'nn_models', datetime.datetime.now().strftime("%Y%m%d-%H%M"))
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # tensorboard --logdir data/tf_logs

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=30, validation_data=(val_data, val_labels), verbose=2,
          callbacks=[modelSaver, tb_cb])

print('\nRelative Head Positions:')
for key in head_loc_counter:
    print(f'{key}: {head_loc_counter[key]/len(labels):.2%}')
