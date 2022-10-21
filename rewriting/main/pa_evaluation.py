import torch
import sys
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.stem import WordNetLemmatizer
import warnings
import logging
from collections import Counter
from sklearn.metrics import accuracy_score

'''
get the power and agency of gernerated text based lexicon 
python pa_evaluation.py <gen_sen_path> <toeval('sen0' or 'out')> 
'''

OUT_PATH = 'pa_eval_res'

# logging
tqdm.pandas()
warnings.filterwarnings('ignore')
logger = logging.getLogger('flair')
logger.setLevel(level=logging.ERROR)
fh = logging.StreamHandler()
logger.addHandler(fh)

# utils
lemmatizer = WordNetLemmatizer()
pos_tagger = SequenceTagger.load("upos-fast")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

# power_agency
pa = pd.read_csv('./CFdata/agency_power_prepro.csv', sep=',')
pa.agency = pa.agency.map({'agency_pos': 'pos', 'agency_equal': 'equal', 'agency_neg': 'neg'}, na_action='ignore')
pa.power = pa.power.map({'power_agent': 'pos', 'power_equal': 'equal', 'power_theme': 'neg'}, na_action='ignore')
pa_verbprep = pa[pa.prep.notna()][['lemma', 'prep']]
pa_verbprep = list(pa_verbprep.itertuples(index=False, name=None))
pa_verblemma = set(pa.lemma)


def cat_from_agency_power(agency, power):
    '''
    create mixed class name from agency and power class names (both in ['pos', 'equal', 'neg']
    '''
    return 'a' + agency + '-' + 'p' + power


def preprocess_texts(text):
    sent = Sentence(text)
    if len(sent) > 0:
        pos_tagger.predict(sent)
        for token in sent:
            if token.get_tag('pos').value == 'VERB':
                token.add_tag('lemma', lemmatizer.lemmatize(token.text, pos='v'))
    return sent


def get_majority(cat_list):
    cat_list = [x for x in cat_list if x != 'nan']
    result = 'nan'
    if len(cat_list) == 1:
        result = cat_list[0]
    elif len(cat_list) > 0:
        # majority
        mc = Counter(cat_list).most_common()
        if len(mc) > 1 and mc[0][1] == mc[1][1]:
            result = 'equal'
        else:
            result = mc[0][0]
    return result


def get_new_category(sent):
    indices = []
    for tid in range(len(sent)):
        token = sent[tid]
        if token.get_tag('pos').value == 'VERB' and token.get_tag('lemma').value in pa_verblemma:
            # verb from pa -> has prep?
            if tid + 1 < len(sent):
                next_token = sent[tid + 1]

                if (token.get_tag('lemma').value, next_token.text) in pa_verbprep:
                    # has prep
                    indices.append(pa[(pa.lemma == token.get_tag('lemma').value) & (pa.prep == next_token.text)].index)
                else:
                    indices.append(pa[(pa.lemma == token.get_tag('lemma').value) & (~pa.prep.notna())].index)
            else:
                index = pa[(pa.lemma == token.get_tag('lemma').value) & (~pa.prep.notna())].index
                if len(index) == 0:
                    index = pa[(pa.lemma == token.get_tag('lemma').value)].index
                indices.append(index)

    # filter indices
    indices = [i[0] for i in indices if len(i) > 0]
    indices = list(set(indices))

    agens = []
    pows = []
    for index in indices:
        agens.append(str(pa.iloc[index].agency))
        pows.append(str(pa.iloc[index].power))

    agency = get_majority(agens)
    power = get_majority(pows)

    return cat_from_agency_power(agency, power)


def compare_cats(descat, newcat, agency=True):
    if agency:
        idx = 0
    else:
        idx = 1
    des = descat.split('-')[idx][1:]
    new = newcat.split('-')[idx][1:]

    if des == 'nan' or new == 'nan':
        return np.nan
    if des == new:
        return 1
    if des == 'equal' or new == 'equal':
        return 0.5
    return 0


def accuracy(df):
    des_a = df.descat.apply(lambda x: x.split('-')[0][1:])
    new_a = df.newcat.apply(lambda x: x.split('-')[0][1:])
    des_p = df.descat.apply(lambda x: x.split('-')[1][1:])
    new_p = df.newcat.apply(lambda x: x.split('-')[1][1:])
    acc = accuracy_score(df.descat, df.newcat)
    acc_a = accuracy_score(des_a, new_a)
    acc_p = accuracy_score(des_p, new_p)
    return acc, acc_a, acc_p


def evaluate_power_agency(df, toeval, write=True):
    df['prepro'] = df[toeval].progress_apply(preprocess_texts)
    df['newcat'] = df.prepro.progress_apply(get_new_category)
    df['asim'] = df.progress_apply(lambda row: compare_cats(row.descat, row.newcat, True), axis=1)
    df['psim'] = df.progress_apply(lambda row: compare_cats(row.descat, row.newcat, False), axis=1)
    df['sim'] = df.progress_apply(lambda row: (row.asim + row.psim) / 2, axis=1)
    # accuracy
    acc, acc_a, acc_p = accuracy(df)

    print(acc, acc_a, acc_p)
    print(df.sim.describe())
    print(df.asim.describe())
    print(df.psim.describe())

    if write:
        # write to file
        with open(OUT_PATH + '/pa_eval.csv', 'a') as f:
            csv_out = csv.writer(f)
            csv_out.writerow((
                sys.argv[1],
                toeval,
                acc,
                acc_a,
                acc_p,
                df.sim.describe()['mean'],
                df.sim.describe()['std'],
                df.asim.describe()['mean'],
                df.asim.describe()['std'],
                df.psim.describe()['mean'],
                df.psim.describe()['std']
            ))
    else:
        return list([
            acc,
            acc_a,
            acc_p,
            df.sim.describe()['mean'],
            df.sim.describe()['std'],
            df.asim.describe()['mean'],
            df.asim.describe()['std'],
            df.psim.describe()['mean'],
            df.psim.describe()['std']
        ])


if __name__ == '__main__':
    filepath = './gen_sen/' + sys.argv[1]
    df = pd.read_csv(filepath)
    toeval = sys.argv[2]

    evaluate_power_agency(df, toeval)
