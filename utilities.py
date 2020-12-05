import string
import os
import xml.etree.cElementTree as cET
import json

json_folder = os.path.join(os.path.expanduser('~'), 'Google Drive', 'PycharmProjects', 'Semantic-Domains')
nv_by_lemma_dict = json.load(open(os.path.join(json_folder, 'nv_l2sd_por.json'), 'r', encoding='utf8'))

pos0_dict = {'a': 'adj', 'n': 'noun', 'v': 'verb', 'd': 'adv', 'c': 'conj', 'g': 'conj', 'r': 'adposition', 'b': 'conj',
             'p': 'pronoun', 'l': 'article', 'i': 'interjection', 'm': 'numeral', 'e': 'interjection'}
pos1_dict = {'1': 'first', '2': 'second', '3': 'third'}
pos2_dict = {'s': 'singular', 'p': 'plural', 'd': 'dual'}
pos3_dict = {'p': 'present', 'i': 'imperfect', 'r': 'perfect', 'a': 'aorist', 'l': 'pluperfect', 'f': 'future', 't':
             'future perfect'}
pos4_dict = {'i': 'indicative', 's': 'subjunctive', 'n': 'infinitive', 'm': 'imperative', 'p': 'participle',
             'o': 'optative'}
pos5_dict = {'a': 'active', 'm': 'middle', 'p': 'passive', 'e': 'middle or passive'}
pos6_dict = {'m': 'masculine', 'f': 'feminine', 'n': 'neuter'}
pos7_dict = {'n': 'nominative', 'g': 'genitive', 'd': 'dative', 'v': 'vocative', 'a': 'accusative'}
agdt2_rel_dict = {'obj': 'object'}
proiel_pos_dict = {'A': 'adj', 'D': 'adv', 'S': 'article', 'M': 'numeral', 'N': 'noun', 'C': 'conj', 'G': 'conj',
                   'P': 'pronoun', 'I': 'interjection', 'R': 'adposition', 'V': 'verb'}


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


# This returns the part-of-speech or the mood if the part-of-speech is a verb for a given word.
def posermooder(f_word):
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 0:
            pos0 = f_word['postag'][0]
            if pos0 in pos0_dict:
                f_pos = pos0_dict[pos0]
                if f_pos == 'verb':
                    if len(f_word['postag']) > 4:
                        pos4 = f_word['postag'][4]
                        if pos4 in pos4_dict:
                            f_pos = pos4_dict[pos4]
            else:
                f_pos = 'other'
        else:
            f_pos = 'other'
    elif f_word.has_attr('part-of-speech'):
        if len(f_word['part-of-speech']) > 0:
            pos0 = f_word['part-of-speech'][0]
            if pos0 in proiel_pos_dict:
                f_pos = proiel_pos_dict[pos0]
                if f_pos == 'verb':
                    if len(f_word['morphology']) > 3:
                        pos3 = f_word['morphology'][3]
                        if pos3 in pos4_dict:
                            f_pos = pos4_dict[pos3]
            else:
                f_pos = 'other'
        else:
            f_pos = 'other'
    else:
        f_pos = 'other'
    return f_pos


# This returns true if pos is a verb.
def verber(f_word):
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 0:
            pos0 = f_word['postag'][0]
            if pos0 in pos0_dict:
                f_pos = pos0_dict[pos0]
                if f_pos == 'verb':
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    elif f_word.has_attr('part-of-speech'):
        if len(f_word['part-of-speech']) > 0:
            pos0 = f_word['part-of-speech'][0]
            if pos0 in proiel_pos_dict:
                f_pos = proiel_pos_dict[pos0]
                if f_pos == 'verb':
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False


# This takes a token or word and returns its number: singular, plural, dual, or other.
def person(f_word):
    f_person = 'other'
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
    return f_person


# This takes a token or word and returns its number: singular, plural, dual, or other.
def grammatical_number(f_word):
    gram_num = 'other'
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
    return gram_num


# This takes a token or word and returns its number: singular, plural, dual, or other.
def tenser(f_word):
    f_tense = 'other'
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
    return f_tense


# This takes a token or word and returns its number: singular, plural, dual, or other.
def mooder(f_word):
    f_mood = 'other'
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
    return f_mood


# This takes a token or word and returns its number: singular, plural, dual, or other.
def voicer(f_word):
    f_voice = 'other'
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
    return f_voice


# This takes a token or word and returns its number: singular, plural, dual, or other.
def gender(f_word):
    f_person = 'other'
    if f_word.has_attr('postag'):
        if len(f_word['postag']) > 6:
            pos6 = f_word['postag'][6]
            if pos6 in pos6_dict:
                f_person = pos6_dict[pos6]
    if f_word.has_attr('morphology'):
        if len(f_word['morphology']) > 5:
            pos6 = f_word['morphology'][5]
            if pos6 in pos6_dict:
                f_person = pos6_dict[pos6]
    return f_person


# This takes a token or word and returns its number: singular, plural, dual, or other.
def caser(f_word):
    f_case = 'other'
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
    return f_case


def deaccent(dastring):
    """Returns an unaccented version of dastring."""
    aeinput = "άἀἁἂἃἄἅἆἇὰάᾀᾁᾂᾃᾄᾅᾆᾇᾰᾱᾲᾳᾴᾶᾷἈἉΆἊἋἌἍἎἏᾈᾉᾊᾋᾌᾍᾎᾏᾸᾹᾺΆᾼέἐἑἒἓἔἕὲέἘἙἚἛἜἝΈῈΈ"
    aeoutput = "ααααααααααααααααααααααααααΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑΑεεεεεεεεεΕΕΕΕΕΕΕΕΕ"
    hoinput = "ΉῊΉῌἨἩἪἫἬἭἮἯᾘᾙᾚᾛᾜᾝᾞᾟήἠἡἢἣἤἥἦἧὴήᾐᾑᾒᾓᾔᾕᾖᾗῂῃῄῆῇὀὁὂὃὄὅόὸόΌὈὉὊὋὌὍῸΌ"
    hooutput = "ΗΗΗΗΗΗΗΗΗΗΗΗΗΗΗΗΗΗΗΗηηηηηηηηηηηηηηηηηηηηηηηηοοοοοοοοοΟΟΟΟΟΟΟΟΟ"
    iuinput = "ΊῘῙῚΊἸἹἺἻἼἽἾἿΪϊίἰἱἲἳἴἵἶἷΐὶίῐῑῒΐῖῗΫΎὙὛὝὟϓϔῨῩῪΎὐὑὒὓὔὕὖὗΰϋύὺύῠῡῢΰῦῧ"
    iuoutput = "ΙΙΙΙΙΙΙΙΙΙΙΙΙΙιιιιιιιιιιιιιιιιιιιΥΥΥΥΥΥΥΥΥΥΥΥυυυυυυυυυυυυυυυυυυυ"
    wrinput = "ώὠὡὢὣὤὥὦὧὼώᾠᾡᾢᾣᾤᾥᾦᾧῲῳῴῶῷΏὨὩὪὫὬὭὮὯᾨᾩᾪᾫᾬᾭᾮᾯῺΏῼῤῥῬ"
    wroutput = "ωωωωωωωωωωωωωωωωωωωωωωωωΩΩΩΩΩΩΩΩΩΩΩΩΩΩΩΩΩΩΩΩρρΡ"
    # Strings to feed into translator tables to remove diacritics.

    aelphas = str.maketrans(aeinput, aeoutput, "⸀⸁⸂⸃·,.—")
    # This table also removes text critical markers and punctuation.

    hoes = str.maketrans(hoinput, hooutput, string.punctuation)
    # Removes other punctuation in case I forgot any.

    ius = str.maketrans(iuinput, iuoutput, '0123456789')
    # Also removes numbers (from verses).

    wros = str.maketrans(wrinput, wroutput, string.ascii_letters)
    # Also removes books names.

    return dastring.translate(aelphas).translate(hoes).translate(ius).translate(wros).lower()


def denumber(dalemma):
    """Removes number from the string dalemma."""

    numers = str.maketrans('', '', '01234567890')

    return dalemma.translate(numers)


def resequence():
    """Numbers each word element in a treebank with a unique sequential id starting from 1. Then adjusts
    head-ids to match the new numbering."""
    os.chdir('/home/chris/Desktop/CustomTB')
    indir = os.listdir('/home/chris/Desktop/CustomTB')
    worddict = {}

    # This will create a dictionary matching old ID's to their new ones so heads can be reassigned
    # then it will assign the new sequential IDs.
    for file_name in indir:
        i = 1
        if not file_name == 'README.md' and not file_name == '.git':
            print(file_name)
            tb = cET.parse(file_name)
            tbroot = tb.getroot()
            if tbroot.tag == 'treebank':
                for body in tbroot:
                    for sentence in body:
                        for word in sentence:
                            if word.tag == 'word':
                                sentenceid = str(sentence.get('id'))
                                wordid = str(word.get('id'))
                                sentwordid = str(sentenceid + '-' + wordid)
                                worddict[sentwordid] = i
                                word.set('id', str(i))
                                i += 1

                # This will assign new head ID's that are in accordance with the new numbering system.
                for body in tbroot:
                    for sentence in body:
                        for word in sentence:
                            if word.tag == 'word':
                                sentenceid = str(sentence.get('id'))
                                headid = str(word.get('head'))
                                sentheadid = str(sentenceid + '-' + headid)
                                if sentheadid in worddict:
                                    newheadid = worddict[sentheadid]
                                    word.set('head', str(newheadid))

                tb.write(file_name, encoding="unicode")
                print("Resequenced:", file_name)

            if tbroot.tag == 'proiel':
                for source in tbroot:
                    for division in source:
                        for sentence in division:
                            for token in sentence:
                                if token.tag == 'token':
                                    sentenceid = str(sentence.get('id'))
                                    wordid = str(token.get('id'))
                                    sentwordid = str(sentenceid + '-' + wordid)
                                    worddict[sentwordid] = i
                                    token.set('id', str(i))
                                    i += 1

                for source in tbroot:
                    for division in source:
                        for sentence in division:
                            for token in sentence:
                                if token.tag == 'token':
                                    sentenceid = str(sentence.get('id'))
                                    headid = str(token.get('head-id'))
                                    sentheadid = str(sentenceid + '-' + headid)
                                    if sentheadid in worddict:
                                        newheadid = worddict[sentheadid]
                                        token.set('head-id', str(newheadid))

                tb.write(file_name, encoding="unicode")
                print("Resequenced:", file_name)


def ider(f_word):
    f_id = False
    if f_word.has_attr('id'):
        f_id = f_word['id']
    return f_id


# This returns the head of the word
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


# Returns a form
def former(f_word):
    if f_word.has_attr('form'):
        return f_word['form']
    else:
        return 'no_form'


# Given a non-verb word, this will return its list of semantic domains (parts-of-speech).
def nv_semdom_poser(f_word):
    semdom_pos_list = []
    pos = poser(f_word)
    if f_word.has_attr('lemma'):
        lemma = deaccent(f_word['lemma']).lower()
        if lemma in nv_by_lemma_dict:
            sem_doms = nv_by_lemma_dict[lemma]
        else:
            sem_doms = ['domain_unknown']
    else:
        sem_doms = ['domain_unknown']
    for domain in sem_doms:
        sem_dom_poss = domain + ' (' + pos + ')'
        semdom_pos_list.append(sem_dom_poss)
    return semdom_pos_list


# This takes a token or word and returns its number: singular, plural, dual, or other.
def lemmer(f_word):
    f_lemma = 'no lemma'
    if f_word.has_attr('lemma'):
        f_lemma = deaccent(f_word['lemma']).lower()
    return f_lemma


# Given the sentence find_all list and a head word, this function returns the words which depend on that head.
def give_dependents(sentence_words, head_word):
    the_words = []
    if head_word.has_attr('id'):
        head_word_id = head_word['id']
        for f_word in sentence_words:
            if f_word.has_attr('head'):
                word_head = f_word['head']
                if word_head == head_word_id:
                    the_words.append(f_word)
            if f_word.has_attr('head-id'):
                word_head = f_word['head-id']
                if word_head == head_word_id:
                    the_words.append(f_word)
    return the_words


# Given a work, this will return a word count which ignores blank words.
def word_count(soup):
    f_count = 0
    f_words = soup.find_all(['word', 'token'])
    for token in f_words:
        if token.has_attr('artificial') or token.has_attr('empty-token-sort'):
            pass
        else:
            f_count += 1
    return f_count

