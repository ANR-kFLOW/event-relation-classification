import re

import nltk
import pandas as pd
from nltk import tokenize

nltk.download('punkt')

sentences = pd.read_csv('/Users/youssrarebboud/Desktop/intention_left.csv')
events = pd.read_csv(
    '/Users/youssrarebboud/PycharmProjects/prompts_based_data_augmentation/left_intention_with_events2.csv')

# sentences.columns=['sentence']
# events.columns=['events']
sentences_prevention = sentences['0'].values
events_prevention = events['generated events'].values

print(sentences.columns)
# print(events_prevention)
pattern = r'(\([a-zA-Z\s]+,\s[a-zA-Z\s]+\)(\n)*)+'
pattern2_enable = r'(\([a-zA-Z]*\)==trigger1\,\s\([a-zA-Z]*\)==trigger2(\n)*)+'
pattern3_enable = r'trigger1:\s[a-zA-Z]+\s\(trigger2:\s[a-zA-Z]+'
pattern4_enable = r'\([a-zA-Z\s-]*\)\s\([a-zA-Z\s-]*\)'


def seperate_sentences(sentences, events):
    sep_sen = []
    evnts = []
    joined = []
    sentences = [re.sub(r'[0-9]+\.', '', k) for k in sentences]
    for sen, evnt in zip(sentences, events):
        sep = tokenize.sent_tokenize(sen)
        a = re.findall(pattern, evnt)
        b = re.findall(pattern2_enable, evnt)
        c = re.findall(pattern3_enable, evnt)
        d = re.findall(pattern4_enable, evnt)
        if a or b or d:
            pairs = []
            if a or b:
                pairs = [k.split(', ') for k in re.sub(r'\(|\)|\n\n|==trigger1|==trigger2', '', evnt).split('\n')]
            elif d:
                pairs = [k.split(')') for k in re.sub(r'\(|\n\n', '', evnt).split('\n')]

            if len(pairs) == len(sep):
                for sn, pr in zip(sep, pairs):
                    joined.append([sn, pr[0], pr[1]])
        elif c:
            if '\n' in evnt:
                pass

            pr = re.sub(r'\(|\)|\n\n|trigger1:|trigger2:', '', evnt).split(' ')
            joined.append([sen, pr[1], pr[3]])

            # pr=re.sub('\(|\)|\n\n|trigger1:|trigger2:', '', evnt).split(' ')

            # joined.append([sen,pr[0],pr[1]])

            #
            #     for sn, pr in zip(sep, pairs):
            #         print(sn, pr)
            #         print(evnt)
            #         print('--------------------')

    return sep_sen, evnts, joined


sep_sen, evnts, joined = seperate_sentences(sentences_prevention, events_prevention)
prv = pd.DataFrame(joined, columns=['sentence', 'trigger1', 'trigger2'])
prv = prv.drop_duplicates()
print(prv)
prv.to_csv('/Users/youssrarebboud/Desktop/cleaned_left_intention.csv')
