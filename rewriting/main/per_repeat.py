import pandas as pd
import sys
'''
repetitiveness: percent greater than <threshold> bigrams in a sentence is <result>
python per_repeat.py <gen_sen_path> <treshold>
'''
def max_num_bigram(sen):
    toks = sen.split()
    if len(toks) < 2:
        return 0
    dct = {}
    for i in range(len(toks) - 1):
        dct[tuple(toks[i:i+1])] = dct.get(tuple(toks[i:i+1]), 0) + 1 # count #occurences of all bigrams
    return max(dct.values()) # max. #occurences of one bigram in a sentence

def get_repetitiveness(df, thres):
    lst = df['out'].tolist()
    len_lst = [max_num_bigram(sen) for sen in lst]
    ori_len = len(len_lst)
    len_lst = list(filter(lambda x: x > thres, len_lst))
    pct = len(len_lst) / ori_len
    print(' percent greater than ', str(thres), 'bigrams in a sentence is ', str(pct))
    return pct


if __name__ == '__main__':
    ds, thres = sys.argv[1], int(sys.argv[2])
    df = pd.read_csv('gen_sen/' + ds + '.csv')
    get_repetitiveness(df, thres)
