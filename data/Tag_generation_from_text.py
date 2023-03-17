# This is a sample Python script.

import re

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import spacy

# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
import pandas as pd

df = pd.read_csv('/Users/youssrarebboud/Desktop/left_intention.csv')
df_filtered = df[df['label'] == 1]
df = df.drop_duplicates().dropna()


def find_sublist_indices(lst, sublst):
    start = 0
    end = 0
    sublist_indices = []
    for i in range(len(lst)):
        if lst[end] == sublst[end - start]:
            end += 1
            if end - start == len(sublst):
                sublist_indices.append((start, end - 1))
                start = end
        else:
            start += 1
            end = start
    return sublist_indices


tags = []
label = []
for index, row in df.iterrows():

    tag_sen = []
    #
    print(row['sentence'])
    #
    event1, event2 = [(re.sub('\.|,', '', token.text)).lower() for token in nlp(row['trigger1'])], [
        (re.sub('\.|,', '', token.text)).lower() for token in nlp(row['trigger2'])]
    print(event1)
    print(event2)
    ind_evnt1 = \
    find_sublist_indices([(re.sub('\.|,', '', token.text)).lower() for token in nlp(row['sentence'])], event1)[0]
    ind_evnt2 = \
    find_sublist_indices([(re.sub('\.|,', '', token.text)).lower() for token in nlp(row['sentence'])], event2)[0]
    evnt1_beg = ind_evnt1[0]
    evnt2_beg = ind_evnt2[0]
    evn1_end = ind_evnt1[1]
    evn2_end = ind_evnt2[1]
    sent = [token.text for token in nlp(row['sentence'])]
    tag2 = 'effect'
    tag1 = 'intention'
    print(evnt1_beg, evn1_end)
    print(evnt2_beg, evn2_end)
    idx = 0
    for word in sent:

        print(sent.index(word))

        if idx >= evnt1_beg and idx <= evn1_end:
            print(word)
            tag_sen.append(tag1)
        elif idx >= evnt2_beg and idx <= evn2_end:
            print(word)
            tag_sen.append(tag2)
        else:
            tag_sen.append('0')
        idx += 1
    tags.append(' '.join(tag_sen))

    label.append('intend')
    print(tag_sen)

df['tag'] = tags
df['label'] = label

print(df)
df.to_csv('/Users/youssrarebboud/Desktop/new_data/intend_left.csv')
#
#
# df_enable1=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/enables_stud.csv')
# df_enable2=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/enables.csv')
#
# df_prevents1=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/prevents_stud.csv')
# df_prevents2=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/prevents.csv')
#
# df_intends1=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/intend_left.csv')
# df_intends2=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/intends.csv')
#
#
# df_previous=pd.read_csv('/Users/youssrarebboud/Desktop/new_data/our_data.csv')
#
# negative_samples=pd.read_csv('/Users/youssrarebboud/Documents/GitHub/Prompts Based Data Augmentation for Event relation extraction/generated_dataset/final cleaned files/joined.csv')
# negative_samples=negative_samples[negative_samples['label'] == '0']
# negative_samples.to_csv('/Users/youssrarebboud/Desktop/new_data/negative_samples.csv')

# texts=list(df_enable1['sentence'].values)+list(df_enable2['sentence'].values)+list(df_prevents1['sentence'].values)+list(df_prevents2['sentence'].values)+list(df_intends1['sentence'].values)+list(df_intends2['sentence'].values)+list(df_previous['sentece'].values)+list(negative_samples['text'].values)
# tags=list(df_enable1['tag'].values)+list(df_enable2['tag'].values)+list(df_prevents1['tag'].values)+list(df_prevents2['tag'].values)+list(df_intends1['tag'].values)+list(df_intends2['tag'].values)+list(df_previous['tag'].values)+list(negative_samples['tag'].values)
# labels=list(df_enable1['label'].values)+list(df_enable2['label'].values)+list(df_prevents1['label'].values)+list(df_prevents2['label'].values)+list(df_intends1['label'].values)+list(df_intends2['label'].values)+list(df_previous['label'].values)+list(negative_samples['label'].values)

# tags=list(df_intends['tag'].values)+list(df_enable['tag'].values)+list(df_prevents['tag'].values)+list(df_previous['tag'].values)
# labels=list(df_intends['label'].values)+list(df_enable['label'].values)+list(df_prevents['label'].values)+list(df_previous['label'].values)
# joined = pd.DataFrame({'text': texts,'tag': tags,'label' :labels})
# print(joined)
# joined.to_csv('/Users/youssrarebboud/Desktop/new_data/new_joined.csv')
