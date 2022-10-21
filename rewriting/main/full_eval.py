from pa_evaluation import *
from bertscore_f1 import *
from gpt2per import *
from per_repeat import *
from datetime import datetime

def map_descat_to_agency(descat):
    return descat.split('-')[0][1:]

def map_descat_to_power(descat):
    return descat.split('-')[1][1:]

if __name__ == '__main__':
    print(datetime.now())

    filepath = './gen_sen/' + sys.argv[1]
    descat = sys.argv[2]
    df = pd.read_csv(filepath, encoding='utf-8')
    df = df[(df.descat == descat) & (df.agency != map_descat_to_agency(descat)) & (df.agency != map_descat_to_power(descat))]
    print('len for', descat, ':', len(df))
    toeval = 'out'
    ref = 'sen'
    thres = 1

    print('PA_EVALUATION')
    pa_eval = evaluate_power_agency(df, toeval, False)

    print('\nBERTScore F1')
    bertscore = measurebsf1(df, ref, toeval)

    print('\nPERPLEXITY')
    perplexity = measurepp(df, toeval)

    print('\nREPETITIVENESS')
    repetitiveness = get_repetitiveness(df, thres)


    with open('full_eval.csv', 'a') as f:
        csv_out = csv.writer(f)
        row = list([sys.argv[1]+'_'+sys.argv[2], toeval, ref])
        # pa_eval
        for i in range(len(pa_eval)):
            row.append(pa_eval[i])
        row.append(bertscore)
        row.append(perplexity[0])
        row.append(perplexity[1])
        row.append(repetitiveness)

        csv_out.writerow(row)

    print(datetime.now())

# python full_eval.py <filename>