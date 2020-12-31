import os
from bs4 import BeautifulSoup
import time
# import tensorflow
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
# 2.1) If the head is elliptical, recognize that. Is training data consistent about pointing articles to ellipses?


corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated')
indir = os.listdir(corpora_folder)
file_count = 0
samples = []
labels = []
total_article_count = 0
replace_works = []
# Search through every work in the annotated Greek folder
for file in indir:
    if file[-4:] == '.xml':
        work_samples = 0
        file_count += 1
        print(file_count, file)
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
                    if lemmer(token) == 'ὁ':
                        total_article_count += 1
                        work_samples += 1
#                        print(f'Article {article_count}: {token["form"]}')
                        window_sequence = []
                        label = []
                        header_tensor = [0] * 15
                        article_index = tokens.index(token)
                        window_start = article_index - 4
                        window_end = article_index + 10
                        while window_start < 0:
                            # print([0] * 49, 'Out of Sentence')
                            window_sequence.append([0] * 49)
                            window_start += 1
                        while window_start < window_end:
                            try:
                                token_tensor = poser(tokens[window_start])[1] + person(tokens[window_start])[1] + \
                                               grammatical_number(tokens[window_start])[1] + \
                                               tenser(tokens[window_start])[1] + mooder(tokens[window_start])[1] + \
                                               voicer(tokens[window_start])[1] + gender(tokens[window_start])[1] + \
                                               caser(tokens[window_start])[1]
                                # print(token_tensor, tokens[window_start]['form'])
                                window_sequence.append(token_tensor)
                            except IndexError:
                                # print([0]*49, 'Out of Sentence')
                                window_sequence.append([0]*49)
                            window_start += 1
                        samples.append(window_sequence)
                        if caser(token) == 'article':
                            label.append([1])
                        else:
                            label.append([0])
                        try:
                            header_index = tokens.index(header(tokens, token))
                            header_window_location = header_index - article_index
                            if -4 <= header_window_location <= 10:
                                header_tensor[header_window_location + 4] = 1
                        except ValueError:
                            pass
                        label.append(header_tensor)
                        labels.append(label)
    print(f'Work Samples/Total Samples: {work_samples}/{total_article_count}')
