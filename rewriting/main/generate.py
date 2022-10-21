import simi_word_vector
from utils_dr import *
from torch import nn
from examples.run_generation import *
import pa_classifier

max_sen_len = 64
random_seed = 7
numepoch = 10
ps = [0.4]
softmax = nn.Softmax(dim=0)
REPEAT_PENALTY = 5


def sample_sequence_ivp(model, length, context, verb_vector, num_samples=1, temperature=1, top_k=0,
                        top_p=0.0, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    orilen = len(context[0])
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)

            # boosting verbs
            verb_vector = verb_vector.to(device)
            next_token_logits += verb_vector

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for j in set(generated[0][orilen + 1:].view(-1).tolist()):
                next_token_logits[j] /= 1

            next_token_logits = softmax(next_token_logits)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.argmax(filtered_logits).unsqueeze(0)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def gen_p(model, test_dataset, descats, verbs, vocabBoost, classifier, sthresh):
    outlist = []
    outp = []

    # get pa boosting vector
    if classifier != 0: # according to classifier
        print('using CLASSIFIER')
        verb_vector_pa = pa_classifier.get_classifier_verb_vector(test_dataset, descats, verbs, classifier, tokenizer_dr, num_added_token_dr, sthresh)
    else: # according to lexicon
        print('using LEXICON')
        verb_vector_pa = create_agen_pow_vector(tokenizer_dr, num_added_token_dr)

    for i in ps:
        for j in trange(len(test_dataset)):
            sen = test_dataset[j]  # e.g. token list of ' <start> bla <VERB> blabla <cls>apos - ppos <start>'
            senlen = len(sen)

            # calc boosting vector
            if classifier != 0:
                vv_pa = verb_vector_pa[j]
            else:
                vv_pa = verb_vector_pa[descats[j]]
            verb_vector = vv_pa * vocabBoost

            out = sample_sequence_ivp(
                model=model,
                context=sen,
                verb_vector=verb_vector,
                length=max_sen_len,
                top_p=i,
                device=device_dr
            )
            out = out[0, senlen:].tolist()
            text = tokenizer_dr.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            end_ind = text.find('<end>')
            if end_ind >= 0:
                text = text[0: end_ind]
            outlist.append(text)
            outp.append(i)
    return outlist, outp


def eval_model(mind, test_dataset, df, vocabBoost, classifier, sthresh, mtd='para'):
    '''
    get generated sentence for a particular model
    '''
    finaldf = pd.DataFrame()
    if mtd == 'para':
        savedir = './modelp/savedmodels'
    elif mtd == 'joint':
        savedir = './modelmix/savedmodels'
    else:
        savedir = './modelr/savedmodels'

    if 'orisen' in df.columns:
        colsen = 'orisen'
    else:
        colsen = 'sen'
    modelpath = savedir + str(mind)
    print('modelpath', modelpath)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device_dr)
    model.eval()
    df = repeatN(df, len(ps) - 1)
    outlist, outp = gen_p(model, test_dataset, df['descat'].tolist(), df['verb'].tolist(), vocabBoost, classifier, sthresh)
    df['out'] = outlist
    df['p-value'] = outp
    df.sort_values(by=[colsen, 'p-value'], inplace=True)
    df['modelind'] = mind
    finaldf = finaldf.append(df, ignore_index=True)
    return finaldf


def gen(mind, ds, vocabBoost, classifier, sthresh, model='para'):
    ds_frac = 1
    if ds == 'roc':
        f = ROC_DEV
        ds_frac = 0.2
    elif ds == 'roc-test':
        f = ROC_TEST
    else:
        print('Please specify dataset!')
        return
    test_dataset, df = parse_file_dr(f, train_time=False, frac=ds_frac)
    print('len(df.index)', len(df.index))
    finaldf = eval_model(mind, test_dataset, df, vocabBoost, classifier, sthresh, mtd=model)

    if classifier != 0:
        savedfile = 'gen_sen/' + model + '-' + ds + '-' + str(mind) + '-' + str(vocabBoost) + '-' + str(classifier) + '-' + str(sthresh) + '.csv'
    else:
        savedfile = 'gen_sen/' + model + '-' + ds + '-' + str(mind) + '-' + str(vocabBoost) + '-' + str(classifier) + '.csv'

    finaldf.to_csv(savedfile, index=False)


def main(ds, mind, mtd, vocabBoost, classifier, sthresh):
    args = {}
    args['n_ctx'] = max_sen_len
    # change to -> load saved dataset
    gen(mind, ds, vocabBoost, classifier, sthresh, mtd)


if __name__ == '__main__':
    # mtd: model trained dataset
    parser = argparse.ArgumentParser(description='Process generation parameters')
    parser.add_argument('--dataset', type=str,
                        help='dataset for generation')
    parser.add_argument('--setup', type=str,
                        help='model setup objective')
    parser.add_argument('--epoch', type=str, default=0,
                        help='the previous trained epoch to load')
    parser.add_argument('--vocabBoost', type=int, default=1,
                        help='the factor for vocabBoosting')
    parser.add_argument('--classifier', type=int, default=0,
                        help='use classifier if != 0; specifies the kind of verb similarity measurement used')
    parser.add_argument('--sthresh', type=float, default=0.5,
                        help='threshold for minimal similarity')
    args = parser.parse_args()
    main(args.dataset, args.epoch, args.setup, args.vocabBoost, args.classifier, args.sthresh)
