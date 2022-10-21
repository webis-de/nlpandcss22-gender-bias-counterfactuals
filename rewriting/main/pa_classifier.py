import torch
from gensim.models import KeyedVectors
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from _transformers import OpenAIGPTTokenizer
from utils_dr import infi2allforms
import pandas as pd
from nltk.corpus import wordnet as wn

CLS_PATH = './pa_classifier/'
AGENCY_CLS_PATH = CLS_PATH + 'agency-bert-base-uncased'
POWER_CLS_PATH = CLS_PATH + 'power-bert-base-uncased'
VERB = '<VERB>'
SEP = '<SEPX>'
GLOVE = 1
WORDNET = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)


def format_input(s, verb):
    # spaces around verb mask?
    idx = s.find(VERB)
    if s[idx+len(VERB)] != ' ': # space after
        s = s[:idx+len(VERB)] + ' ' + s[idx+len(VERB):]
    if s[idx-1] != ' ': # space before
        s = s[:idx] + ' ' + s[idx:]

    return verb + ' ' + SEP + ' ' + s


def format_target_class(descat):
    split = descat.split('-')
    result = [0, 0]
    for i in range(len(split)):
        if 'pos' in split[i]:
            result[i] = 0
        elif 'equal' in split[i]:
            result[i] = 1
        elif 'neg' in split[i]:
            result[i] = 2
        else:
            print('ERROR: descat', descat, 'unknown')
    return result[0], result[1]


def read_model(save_dir):
    model = BertForSequenceClassification.from_pretrained(save_dir, num_labels=3)
    model.to(device)
    return model


def pred_batch(model, input_list, batch_size, tokenizer):
    predictions = []
    i1 = 0
    i2 = batch_size
    with tqdm(total=len(input_list)) as pbar:
        while i1 < len(input_list):
            sequences = input_list[i1:i2]
            batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**batch.to(device))
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

            i1 += batch_size
            i2 += batch_size
            if i2 > len(input_list):
                i2 = len(input_list)
            pbar.update(batch_size)
    return predictions


def prepare_s(s, tokenizer_trans):
    s = tokenizer_trans.decode(s)
    s = s.replace('<start>', '')
    s_split = s.split('<cls>')
    return s_split[0].strip()


def get_simi_words_glove(all_verbs, verb, glove_model, thresh=0.):
    if verb not in glove_model:
        return []

    all_verbs = [v for v in all_verbs if v != verb and isinstance(v, str) and v in glove_model]
    return [v for v in all_verbs if glove_model.similarity(verb, v) > thresh]


def get_wordnet_synset(verb):
    ss = wn.synsets(verb, pos=wn.VERB)
    if len(ss) == 0:
        return None
    else:
        return ss[0]


def rescale(x):
    return (2 * (x-0)/(1-0)) -1


def get_simi_words_wordnet(all_verbs, verb, thresh=0.):
    verb_synset = get_wordnet_synset(verb)
    if not verb_synset:
        return []

    all_verbs = [v for v in all_verbs if v != verb and get_wordnet_synset(v)]
    all_verbs_ss = [get_wordnet_synset(v) for v in all_verbs]
    return [all_verbs[i] for i in range(len(all_verbs)) if rescale(verb_synset.wup_similarity(all_verbs_ss[i])) > thresh]


def get_classifier_verb_vector(slist, descats, verbs, simi, tokenizer_trans, num_added, simi_thresh):
    # handle input
    slist = [prepare_s(s, tokenizer_trans) for s in slist]

    # all verbs
    all_verbs = verb_form[0]
    all_verbs = [v for v in all_verbs if isinstance(v, str)]

    # considered verbs (only verbs with positive similarity to verb in sentence)
    if simi == WORDNET:
        print('using WORDNET for simi')
        cons_verbs = [get_simi_words_wordnet(all_verbs, verbs[i], simi_thresh) for i in range(len(slist))]
    else:
        print('using GLOVE for simi')
        glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
        cons_verbs = [get_simi_words_glove(all_verbs, verbs[i], glove_model, simi_thresh) for i in range(len(slist))]
    print(f'{sum([len(cv) for cv in cons_verbs])/len(cons_verbs)} verbs found on average')

    # predict agency + power for verbs in sentences
    inputs = [format_input(slist[i], v) for i in range(len(slist)) for v in cons_verbs[i]] # iterates over verbs first (1. sentence + all verbs, 2. sentence + all verbs,... )
    act_agencies = pred_batch(agency_model, inputs, 1000, tokenizer_cls)
    act_powers = pred_batch(power_model, inputs, 1000, tokenizer_cls)

    # group for input sentences (cls preds for cons_verbs)
    act_a_grouped = []
    i1, i2 = 0, 0
    for i in range(len(slist)):
        i2 = i2 + len(cons_verbs[i])
        act_a_grouped.append(act_agencies[i1:i2])
        i1 = i2

    act_p_grouped = []
    i1, i2 = 0, 0
    for i in range(len(slist)):
        i2 = i2 + len(cons_verbs[i])
        act_p_grouped.append(act_powers[i1:i2])
        i1 = i2

    # construct output vectors
    vectors = [torch.zeros(tokenizer_trans.vocab_size + num_added) for _ in range(len(slist))]
    # get indices with 1
    for i in range(len(slist)):
        aa = act_a_grouped[i]
        ap = act_p_grouped[i]
        cv = cons_verbs[i]
        des_agency, des_power = format_target_class(descats[i])
        idx = [j for j in range(len(cv)) if aa[j] == des_agency and ap[j] == des_power]

        forms = [infi2allforms(cv[i]) for i in idx]
        forms = [item for sublist in forms for item in sublist] # flatten
        for form in forms:
            v_li = tokenizer_trans.encode(form)
            vectors[i][v_li[0]] = 1
    return vectors

verb_form = pd.read_csv('verb.txt', usecols=[_ for _ in range(24)], header=None)

# init models
agency_model = read_model(AGENCY_CLS_PATH)
agency_model.eval()
power_model = read_model(POWER_CLS_PATH)
power_model.eval()

tokenizer_cls = BertTokenizer.from_pretrained('bert-base-uncased')
num_added_tokens = tokenizer_cls.add_special_tokens({
    'additional_special_tokens': [VERB, SEP]
})


# TEST
#tokenizer_trans = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
#get_classifier_verb_vector(' <start>charlie <VERB>out and leaves to an unknown fate, disappearing into the mist. <cls>apos - ppos <start>', 'apos-ppos', tokenizer_trans, 0)
