import os
from bs4 import BeautifulSoup
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter


# Create a custom model saver
class ModelSaver(tf.keras.callbacks.Callback):
    # A custom tensorflow model saver that returns useful information
    very_best = 100
    best_loss = 100
    best_acc = 0
    best_val_acc = 0
    prev_loss = 0
    rep_loss = 0
    best_epoch = 0
    best_mod = 0
    best_units = 0
    best_lr = 0
    best_dropout = 0
    new_best = False

    def on_train_begin(self, logs=None):
        self.best_loss = 100
        self.best_acc = 0
        self.best_val_acc = 0
        self.new_best = False

    def on_epoch_end(self, epoch, logs=None):
        # Save the best model
        # if logs['loss'] < self.best_loss:
        #     self.best_loss = logs['loss']
        #     model_name = "m{0:.3e}val{1:.3e}.m5".format(logs['loss'], logs['val_loss'])
        #     self.model.save(model_name, overwrite=True)
        #     self.best_epoch = epoch + 1
        #     print('\n\nModel saved at epoch', epoch + 1, 'with', self.very_best, 'loss.\n')
        # print(logs.keys())
        if logs['val_accuracy'] > self.best_val_acc:
            self.best_val_acc = logs['val_accuracy']
            # os.path.join('data', 'tf_logs', "m{0:.3f}val{1:.3f}".format(logs['accuracy'], logs['val_accuracy']))
            model_name = os.path.join('data', 'tf_logs', "m{0:.3f}val{1:.3f}".format(logs['accuracy'],
                                                                                     logs['val_accuracy']))
            tf.saved_model.save(model, model_name)
            self.best_epoch = epoch + 1
            self.new_best = True
            print('\n\nModel saved at epoch', epoch + 1, 'with', self.best_val_acc, 'validation accuracy.\n')

    def on_train_end(self, logs=None):
        if self.new_best:
            print('\nBest Model saved at epoch', self.best_epoch, 'with', self.best_val_acc, 'validation accuracy.')


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


def header(f_sentence, f_word):
    """Returns the token's head."""
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


def poser(f_word):
    """Returns the part-of-speech of a token. Participles are considered verbs."""
    f_pos = 'other'
    poses = ('adj', 'adposition', 'adv', 'article', 'conj', 'interjection', 'noun', 'numeral', 'other', 'pronoun',
             'verb')
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
    pos_tensor = [0]*11
    pos_tensor[poses.index(f_pos)] = 1
    return f_pos, pos_tensor


def person(f_word):
    """Returns a token's person: first, second, third, or other."""
    f_person = 'other'
    persons = ('first', 'second', 'third', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 1:
            pos1 = f_word['postag'][1]
            if pos1 in pos1_dict:
                f_person = pos1_dict[pos1]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 0:
            pos1 = f_word['morphology'][0]
            if pos1 in pos1_dict:
                f_person = pos1_dict[pos1]
    person_tensor = [0]*4
    person_tensor[persons.index(f_person)] = 1
    return f_person, person_tensor


def grammatical_number(f_word):
    """Returns a token's grammatical number: singular, plural, dual, or other."""
    gram_num = 'other'
    numbers = ('singular', 'plural', 'dual', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 2:
            pos2 = f_word['postag'][2]
            if pos2 in pos2_dict:
                gram_num = pos2_dict[pos2]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 1:
            pos1 = f_word['morphology'][1]
            if pos1 in pos2_dict:
                gram_num = pos2_dict[pos1]
    number_tensor = [0]*4
    number_tensor[numbers.index(gram_num)] = 1
    return gram_num, number_tensor


def tenser(f_word):
    """Return a token's tense. Return 'other' if not a verb."""
    f_tense = 'other'
    tenses = ('aorist', 'future', 'future perfect', 'imperfect', 'perfect', 'pluperfect', 'present', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 3:
            pos3 = f_word['postag'][3]
            if pos3 in pos3_dict:
                f_tense = pos3_dict[pos3]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 2:
            pos3 = f_word['morphology'][2]
            if pos3 in pos3_dict:
                f_tense = pos3_dict[pos3]
    tense_tensor = [0]*8
    tense_tensor[tenses.index(f_tense)] = 1
    return f_tense, tense_tensor


def mooder(f_word):
    """Returns a token's mood. Returns 'other' if not a verb. Participles are considered a mood here."""
    f_mood = 'other'
    moods = ('indicative', 'subjunctive', 'infinitive', 'imperative', 'participle', 'optative', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 4:
            pos4 = f_word['postag'][4]
            if pos4 in pos4_dict:
                f_mood = pos4_dict[pos4]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 3:
            pos4 = f_word['morphology'][3]
            if pos4 in pos4_dict:
                f_mood = pos4_dict[pos4]
    mood_tensor = [0]*7
    mood_tensor[moods.index(f_mood)] = 1
    return f_mood, mood_tensor


def voicer(f_word):
    """Returns a token's voice. Returns 'other' if not a verb."""
    f_voice = 'other'
    voices = ('active', 'middle', 'passive', 'middle or passive', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 5:
            pos5 = f_word['postag'][5]
            if pos5 in pos5_dict:
                f_voice = pos5_dict[pos5]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 4:
            pos5 = f_word['morphology'][4]
            if pos5 in pos5_dict:
                f_voice = pos5_dict[pos5]
    voice_tensor = [0]*5
    voice_tensor[voices.index(f_voice)] = 1
    return f_voice, voice_tensor


def gender(f_word):
    """Returns a token's gender. Returns 'other' if a verb."""
    f_gender = 'other'
    genders = ('masculine', 'feminine', 'neuter', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 6:
            pos6 = f_word['postag'][6]
            if pos6 in pos6_dict:
                f_gender = pos6_dict[pos6]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 5:
            pos6 = f_word['morphology'][5]
            if pos6 in pos6_dict:
                f_gender = pos6_dict[pos6]
    gender_tensor = [0]*4
    gender_tensor[genders.index(f_gender)] = 1
    return f_gender, gender_tensor


def caser(f_word):
    """Returns a token's case. Returns 'other' if it is a non-participle verb."""
    f_case = 'other'
    cases = ('nominative', 'genitive', 'dative', 'accusative', 'vocative', 'other')
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 7:
            pos7 = f_word['postag'][7]
            if pos7 in pos7_dict:
                f_case = pos7_dict[pos7]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 6:
            pos6 = f_word['morphology'][6]
            if pos6 in pos7_dict:
                f_case = pos7_dict[pos6]
    case_tensor = [0]*6
    case_tensor[cases.index(f_case)] = 1
    return f_case, case_tensor


def lemmer(f_word):
    """Returns the token's lemma. If attribute is missing returns 'missing'."""
    try:
        return f_word['lemma']
    except KeyError:
        return 'missing'

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
pos_labels = []
head_labels = []
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
                        if poser(token)[0] not in ['article', 'pronoun']:
                            print(f'Problem! {file}, sentence {sentence["id"]} token {token["id"]}.')
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
                        # Create the first classification label: identifies the ὁ as an article or not an article
                        if caser(token) == 'article':
                            article_status = [1]
                        else:
                            article_status = [0]
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
                        pos_labels.append(article_status)
                        head_labels.append(head_tensor)
        print(f'Work Samples/Total Samples: {work_samples}/{total_samples_count}')
        print(f'Percent of Samples in Work are Articles: {(article_count/work_samples):.02%}')
print(f'Part-of-Speech of instances of ὁ: {pos_counter}')
print(head_loc_counter)

# Split data into an 80%/20% training/validation split.
split = int(.8*len(head_labels))
train_data = samples[:split]
val_data = samples[split:]
train_labels = head_labels[:split]
val_labels = head_labels[split:]

# Enter the samples and labels into Tensorflow to train a neural network
model = tf.keras.Sequential()
# Input_shape = (n_steps, n_features). Since I combined everything into one tensor, try 49 features for now.
model.add(layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(15, 49)))
model.add(layers.Dense(16, activation='softmax'))
modelSaver = ModelSaver()

log_dir = "data\\tf_logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
# log_dir = os.path.join('data', 'nn_models', datetime.datetime.now().strftime("%Y%m%d-%H%M"))
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # tensorboard --logdir data/tf_logs

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=50, validation_data=(val_data, val_labels), verbose=2,
          callbacks=[modelSaver, tb_cb])

print('\nRelative Head Positions:')
for key in head_loc_counter:
    print(f'{key}: {head_loc_counter[key]/len(head_labels):.2%}')
