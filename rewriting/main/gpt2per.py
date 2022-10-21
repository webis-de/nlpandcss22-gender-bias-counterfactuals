from tqdm import trange
from _transformers import *
import torch
import sys
import pandas as pd
import math
'''
get the perplexity on gpt2
python gpt2per.py <dataset> <method> <toeval('sen' or 'sen0' or 'out')> <gen_sen_path>
'''
device_2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def measurepp(df, toeval):
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)
    outsen = df[toeval].tolist()
    gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2.eval()
    gpt2.to(device_2)
    res = 0
    count = 0
    for i in trange(len(outsen)):
        sen = outsen[i]
        tok_li = gpt2tokenizer.encode(sen, add_special_tokens=False)
        tok_ten = torch.tensor(tok_li, device=device_2)
        loss, logits, past = gpt2(tok_ten, labels=tok_ten)
        if not loss.isnan(): #todo rausnehmen?
            count += 1
            res += loss
    res /= count
    print(res, math.exp(res), count)
    return list([res, math.exp(res)])

if __name__ == '__main__':
    ds, method = sys.argv[1], sys.argv[2]
    filepath = './gen_sen/' + sys.argv[4]
    df = pd.read_csv(filepath)
    toeval = sys.argv[3]
    if ds == 'para':
        df = df.sample(n=1000)
    measurepp(df, toeval)
