import pandas as pd
from nltk.stem import WordNetLemmatizer 
from _transformers import *

def gen_cri(df):
    df['cri'] = df['sen'].str.replace(' ', '')
    df['cri'] = df['cri'].str.replace('[^\w\s]', '')
    df['cri'] = df['cri'].str.lower()
    return df

def dev_he(df):
    hedf = pd.read_csv(ROC_DEV_HE)
    df = gen_cri(df)
    hedf = gen_cri(hedf)
    df = df[df['cri'].isin(hedf['cri'])]
    df.drop(columns=['cri'], inplace=True)
    return df

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list


def agen_pow_verbs():
    a_verbs = agen_verbs()
    p_verbs = pow_verbs()
    pa_verbs = {}

    for ka, va in a_verbs.items():
        for kp, vp in p_verbs.items():
            new_key = 'a' + ka + '-' + 'p' + kp
            common_elements = list(set(va).intersection(set(vp)))
            pa_verbs[new_key] = common_elements

    return pa_verbs


def agen_verbs():
    '''
    for word in each category, get its infinitive form if it's in verb.txt
    for short phrases like 'apply to', only the first word is considered
    Note: 24 words not in verb.txt
    '''
    df = pd.read_csv('./CFdata/agency_verb.csv')
    agen_v = {}
    total = 0
    cats = {'+': 'pos', '-':'neg', '=':'equal'}
    for k, v in cats.items():
        subdf = df[df['Agency{agent}_Label'] == k]
        ver_li = subdf['verb'].str.split()
        agen_v[v] = set(word_infinitive(li[0]) for li in ver_li if len(li) > 0)
        total += len(agen_v[v])
    return agen_v

def pow_verbs():
    '''
    for word in each category, get its infinitive form if it's in verb.txt
    for short phrases like 'apply to', only the first word is considered
    Note: 24 words not in verb.txt
    '''
    df = pd.read_csv('./CFdata/agency_power_prepro.csv')
    power_v = {}
    total = 0
    cats = {'power_agent':'pos', 'power_theme':'neg', 'power_equal':'equal'}
    for k, v in cats.items():
        subdf = df[df['power'] == k]
        ver_li = subdf['verb'].str.split()
        power_v[v] = set(word_infinitive(li[0]) for li in ver_li if len(li) > 0)
        total += len(power_v[v])
    return power_v

def word_infinitive(word):
    #infi = lemmatizer.lemmatize(word)
    row = verb_form[verb_form.isin([word]).any(axis=1)]
    if row.empty:
        return word
    infi = row[0].iloc[0]
    return infi 

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def add_pad(list, tokenizer):
    res = [__sen_pad(sen, tokenizer) for sen in list]
    return res

def __sen_pad(sen, tokenizer):
    # add padding for each sentence
    if len(sen) < max_sen_len:
        pad = [tokenizer.pad_token_id for i in range(max_sen_len - len(sen))]
        sen.extend(pad)
        return sen
    elif len(sen) > max_sen_len:
        orilen = len(sen)
        for i in range(orilen - max_sen_len):
            sen.pop(len(sen) - 2)
    return sen


max_sen_len = 64
#lemmatizer = WordNetLemmatizer() 
verb_form = pd.read_csv('verb.txt', usecols=[_ for _ in range(24)], header=None)
ps = [0.4, 0.6]
num_epoch = 10

agen_pow_v = agen_pow_verbs()

PREFIX = './data/movie_data/'
ROC_TRAIN = PREFIX + 'movie_train.csv'
ROC_TEST = PREFIX + 'movie_test.csv'
ROC_TEST_HE = PREFIX + 'supplyVerb.csv'
ROC_DEV = PREFIX + 'movie_dev.csv'
ROC_DEV_HE = PREFIX + 'for_human_eval.csv'
