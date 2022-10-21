import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet as wn
import torch
from utils_dr import infi2allforms

GLOVE = 1
WORDNET = 2
pa_cats = ['apos-ppos', 'apos-pequal', 'apos-pneg',
           'aequal-ppos', 'aequal-pequal', 'aequal-pneg',
           'aneg-ppos', 'aneg-pequal', 'aneg-pneg']

verb_form = pd.read_csv('verb.txt', usecols=[_ for _ in range(24)], header=None)

def get_simi_glove(verb, tokenizer_trans, num_added, glove_model):
    vector = torch.zeros(tokenizer_trans.vocab_size + num_added)
    if verb not in glove_model:
        return vector

    for v in verb_form[0]: # iterate over all ~8.000 verbs (infinitive)
        if verb != v and isinstance(v, str) and v in glove_model:
            simi = glove_model.similarity(verb, v) # [-1,1]
            if simi > 0:
                # set vector to simi with verb
                forms = infi2allforms(v)
                for form in forms:
                    v_li = tokenizer_trans.encode(form)
                    vector[v_li[0]] = torch.from_numpy(np.asarray(simi))
    return vector


def get_wordnet_synset(verb):
    ss = wn.synsets(verb, pos=wn.VERB)
    if len(ss) == 0:
        return None
    else:
        return ss[0]

def rescale(x):
    return (2 * (x-0)/(1-0)) -1

def get_simi_wordnet(verb, tokenizer_trans, num_added):
    verb_synset = get_wordnet_synset(verb)
    vector = torch.zeros(tokenizer_trans.vocab_size + num_added)
    if not verb_synset:
        return vector

    for v in verb_form[0]: # iterate over all ~8.000 verbs (infinitive)
        v_synset = get_wordnet_synset(v)
        if verb != v and v_synset:
            simi = rescale(verb_synset.wup_similarity(v_synset)) # [0,1] -> [-1,1]
            if simi > 0:
                # set vector to simi with verb
                forms = infi2allforms(v)
                for form in forms:
                    v_li = tokenizer_trans.encode(form)
                    vector[v_li[0]] = torch.from_numpy(np.asarray(simi))
    return vector


def get_simi_word_vector(verb, tokenizer_trans, num_added, source=GLOVE):
    if source == GLOVE:
        glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
        return get_simi_glove(verb, tokenizer_trans, num_added, glove_model)
    elif source == WORDNET:
        return get_simi_wordnet(verb, tokenizer_trans, num_added)


# TEST
#tokenizer_trans = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
#get_simi_word_vector('apply', tokenizer_trans, 0, WORDNET)