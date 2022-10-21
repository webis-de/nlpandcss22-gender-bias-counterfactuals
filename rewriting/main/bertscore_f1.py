from datasets import load_metric
import torch
import sys
import pandas as pd
import subprocess
'''
get the bertscore
python bertscore_f1.py <gen_sen_path> <ref('sen' or 'sen0')> <toeval('sen0' or 'out')> 
'''
device_2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def measurebsf1(df, ref, toeval):
    # prepare data
    path_ref = 'bertscore_temp/refs.txt'
    path_out = 'bertscore_temp/out.txt'

    ref_list = df[ref].tolist()
    with open(path_ref, 'w+', encoding='utf-8') as f:
        for item in ref_list:
            f.write("%s\n" % item)

    out_list = df[toeval].tolist()
    with open(path_out, 'w+', encoding='utf-8') as f:
        for item in out_list:

            f.write("%s\n" % item)
    # call bert-score
    score = subprocess.check_output(['bert-score', '-r', path_ref, '-c', path_out, '--lang', 'en'])
    print(score)
    return score



if __name__ == '__main__':
    filepath = './gen_sen/' + sys.argv[1]
    df = pd.read_csv(filepath)
    ref = sys.argv[2]
    toeval = sys.argv[3]

    measurebsf1(df, ref, toeval)
