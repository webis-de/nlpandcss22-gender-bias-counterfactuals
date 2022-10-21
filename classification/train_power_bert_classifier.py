#!/usr/bin/env python
# coding: utf-8

# train final model on complete training dataset

from datetime import datetime
print(datetime.now())


# In[2]:

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pickle
import random
import csv
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

import logging
logger = logging.getLogger('transformers')
logger.setLevel(logging.ERROR)
fh = logging.StreamHandler()
logger.addHandler(fh)


PREFIX = "../../data/"
PA_PATH = PREFIX + "sap2017-connotation-frames-power-agency/"
TT_PATH = PA_PATH + "train-test-splits/"
MODEL_PATH = 'power_bert_context_models_v3/'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

# # prepare train data

with open(TT_PATH + 'train_power_vp.pickle', 'rb') as f:
    train_vp = pickle.load(f)

pa = pd.read_pickle(PA_PATH + 'power_agency_sents.pkl')
pa = pa[pa.verb_prep.isin(train_vp)]

pa['sents_raw'] = pa.sents.progress_apply(lambda slist : [s.to_plain_string() for s in slist])

MASK = '<VERB>'
SEP = '<SEPX>'
def get_mask_w_context(row):
    result = []
    for sent in row.sents:
        string = sent.to_plain_string()
        found = False

        for token in sent:
            if not found and token.get_tag('lemma').value == row.lemma:
                string = token.text + ' ' + SEP + ' ' + string.replace(token.text, MASK) 
                found = True
        result.append(string)
    return result

pa['sents_masked'] = pa.progress_apply(get_mask_w_context, axis=1)


# ### balance data

def balance_power(row, col):   
    '''balances the power training examples'''
    
    if row.power == 'power_agent': 
        row[col] = random.Random(42).sample(row[col], 34)
    if row.power == 'power_equal': 
        row[col] = random.Random(42).sample(row[col], 134)
    return row

pa = pa.progress_apply(lambda row: balance_power(row, 'sents_masked'), axis=1)
pa = pa.drop('index',axis=1)
pa.reset_index(inplace=True)


X = pa.sents_masked
y = pa.power.map({'power_agent': 0, 'power_equal': 1, 'power_theme': 2 })

# ## train + test

def train(model, X, y, epochs, batch_size, learning_rate):
    for e in range(epochs):
        i1 = 0
        i2 = batch_size
        while i1 < len(X):
            sequences = X[i1:i2]
            batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
            batch["labels"] = torch.tensor(np.asfarray(y).astype(int)[i1:i2]).long()

            optimizer = AdamW(model.parameters(), lr=learning_rate)
            loss = model(**batch.to(device)).loss
            loss.backward()
            optimizer.step()

            i1 += batch_size
            i2 += batch_size
            if i2 > len(X):
                i2 = len(X)
        print(f'epoch {e+1}/{epochs}: {loss}')

def read_model(save_dir):
    model = BertForSequenceClassification.from_pretrained(save_dir, num_labels=3)
    model.to(device)
    return model

# ## train and save

target='power'
emb='sents_masked'


final_setting = [
    ("bert-base-uncased",12,20,1e-08)
]


for checkpoint, epoch, batch_size, learning_rate in final_setting:
    print(checkpoint, epoch, batch_size, learning_rate)

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    num_added_tokens = tokenizer.add_special_tokens({
        'additional_special_tokens': [MASK, SEP]
    })

    save_dir = MODEL_PATH \
           + checkpoint + ' ' \
           + str(epoch) + ' ' \
           + str(batch_size) + ' ' \
           + str(learning_rate)
    print('save_dir', save_dir)

    # check if exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Xy = pd.concat([X, y], axis=1)
    Xy = Xy.explode(emb)

    # fresh model
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    model.resize_token_embeddings(tokenizer.vocab_size + num_added_tokens)
    model.to(device)

    # get X's
    X_train = Xy[emb]
    X_train = X_train.values.tolist()

    # get y's
    y_train = Xy[target]

    # random shuffle train set
    Xy_train = list(zip(X_train, y_train))
    random.Random(42).shuffle(Xy_train)
    X_train, y_train = zip(*Xy_train)

    # train
    train(model, X_train, y_train, epoch, batch_size, learning_rate)

    # save
    model.save_pretrained(save_dir)

    print('saved', save_dir)