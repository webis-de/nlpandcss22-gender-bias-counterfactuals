from utils import *
from gensim.models.keyedvectors import KeyedVectors
import os
import sys
from nltk.corpus import wordnet as wn

### !! not in use, but not fully removed from code !!

#fw = open('verb_outside_glove.txt', 'w')
GLOVE = 1
WORDNET = 2
pa_cats = ['apos-ppos', 'apos-pequal', 'apos-pneg',
           'aequal-ppos', 'aequal-pequal', 'aequal-pneg',
           'aneg-ppos', 'aneg-pequal', 'aneg-pneg']

# *** GLOVE ***
def simi_each_word_glove(verb, agen_pow_v, vcat, glove_model, pa_cats):
    res = []
    res.append(verb)
    res.append(vcat)
    if verb not in glove_model:
        res.append('none')
        res.append('none')
        res.append('none')
        return res
    for cat in pa_cats:
        verbset = agen_pow_v[cat]
        '''
        for v in verbset:
            if v not in glove_model:
                fw.write(v+'\n')
        '''
        verbset = list(filter(lambda v: v in glove_model, verbset)) 
        cp_vs = verbset.copy()
        if verb in cp_vs:
            cp_vs.remove(verb)
        verb_simi = glove_model.most_similar_to_given(verb, cp_vs)
        res.append(verb_simi)
    return res

def simi_verb_each_cat_glove():
    '''
    for each word, get its most similar word in each cat from glove emb
    both word and its simi words are supposed to be in infinitive form
    '''
    # cat -> infi form of words
    df = pd.DataFrame()
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
    vs_col = ['verb', 'oricat']
    vs_col.extend(pa_cats)
    for cat, verbset in agen_pow_v.items():
        data = [simi_each_word_glove(v, agen_pow_v, cat, glove_model, pa_cats) for v in verbset]
        catdf = pd.DataFrame(data, columns=vs_col)
        df = df.append(catdf)
    df.to_csv('verb2simiGLOVE.csv')
    return df

# *** WORDNET ***
def get_synset(verb):
    ss = wn.synsets(verb, pos=wn.VERB)
    if len(ss) == 0:
        return None
    else:
        return ss[0]

def simi_each_word_wordnet(verb, agen_pow_v, vcat, pa_cats):
    res = []
    res.append(verb)
    res.append(vcat)

    verb_synset = get_synset(verb)
    if not verb_synset:
        for i in range(len(pa_cats)):
            res.append('none')
        return res

    for cat in pa_cats:
        verbset = agen_pow_v[cat]
        cp_vs = verbset.copy()
        if verb in cp_vs:
            cp_vs.remove(verb)

        # get most similar verb from list
        # https://subscription.packtpub.com/book/application-development/9781782167853/1/ch01lvl1sec16/calculating-wordnet-synset-similarity
        vs_synsets = [(get_synset(v),v) for v in cp_vs]
        vs_simis = [(verb_synset.lch_similarity(ss[0]), ss[1]) for ss in vs_synsets if ss[0]]
        vs_simis.sort(key=lambda x: x[0], reverse=True)
        verb_simi = vs_simis[0][1]

        res.append(verb_simi)
    return res

def simi_verb_each_cat_wordnet():
    '''
    for each word, get its most similar word in each cat from wordnet dictionary
    both word and its simi words are supposed to be in infinitive form
    '''
    # cat -> infi form of words
    df = pd.DataFrame()
    vs_col = ['verb', 'oricat']
    vs_col.extend(pa_cats)
    for cat, verbset in agen_pow_v.items():
        data = [simi_each_word_wordnet(v, agen_pow_v, cat, pa_cats) for v in verbset]
        catdf = pd.DataFrame(data, columns=vs_col)
        df = df.append(catdf)
    df.to_csv('verb2simiWORDNET.csv')
    return df



def load_word2simi(source):
    '''
    return the dataframe with column [,verb,oricat,apos-ppos,apos-pequal,apos-pneg,aequal-ppos,aequal-pequal,aequal-pneg,aneg-ppos,aneg-pequal,aneg-pneg]
    '''
    if source == GLOVE:
        print('GLOVE')
        csvfile = './verb2simiGLOVE.csv'
    elif source == WORDNET:
        print('WORDNET')
        csvfile = './verb2simiWORDNET.csv'
    else:
        print('NO WORD2SIMI SPECIFIED! ERROR')

    if os.path.exists(csvfile):
        print('reading verb2simi from csv')
        return pd.read_csv(csvfile)
    elif source == GLOVE:
        return simi_verb_each_cat_glove()
    elif source == WORDNET:
        return simi_verb_each_cat_wordnet()