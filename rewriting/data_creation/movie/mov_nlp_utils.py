import xmltodict
import gzip

PREFIX = "../data/"
MS_PATH = PREFIX + "bamman2013-movie-summaries/MovieSummaries/"
NLP_PATH = MS_PATH + 'corenlp_plot_summaries/'


def get_nlp_file(mid):
    with gzip.open(NLP_PATH + str(mid) + '.xml.gz', 'rb') as f:
        content = xmltodict.parse(f.read())['root']['document']
    if 'sentences' in content.keys():
        sents = content['sentences']
    else:
        sents = None
    if 'coreference' in content.keys():
        coref = content['coreference']
    else:
        coref = None
    return sents, coref

def get_sentence(sentences, s_id):
    return [s for s in sentences['sentence'] if s['@id'] == s_id][0]

def get_token(sentences, s_id, t_id):
    return [t for t in get_sentence(sentences, s_id)['tokens']['token'] if t['@id'] == t_id][0]

def get_token_for_sentence(sentence, t_id):
    return [t for t in sentence['tokens']['token'] if t['@id'] == t_id][0]

def sentence_to_raw_string(sent):
    result = ' '.join([t['word'] for t in sent['tokens']['token'] if not t['word'] in ['-LRB-', '-LCB-', '-RCB-', '-RRB-']])
    return result.strip()