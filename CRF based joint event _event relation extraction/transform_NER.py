# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import spacy
# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

import os
import pandas as pd
from main import *
from read_data import *
# def get_pos_data(sentence):
#     doc=nlp(sentence)
#     tags=[token.pos_ for token in doc]
#     return tags
# our_annotations_path = [timebank_path, event_causality_path]
# data = []
# psd_tag_data=[]
# text=[]
# cpt = 1
# lab=[]
# for path in our_annotations_path:
#     for file in os.listdir(path):
#         print(file)
#
#         if file != '.DS_Store':
#
#             # label = 1 if file[:-4] == 'causes' else 2 if file[:-4] =='enables' else 3 if file[:-4]== 'prevents' else 4 if file[:-4] =='intends' else 5
#             df = read_file(path + '/' + file)
#
#             for index, row in df.iterrows():
#                 event1, event2 = get_events(row)
#                 # data.append(row['sentence'])
#                 # sent = row['sentence'].split()
#                 text.append(row['sentence'])
#                 sent=[token.text for token in nlp(row['sentence'])]
#                 tag2 = 'effect'
#                 tag1 = 'cause' if file[:-4] == 'causes' else 'condition' if file[
#                                                                             :-4] == 'enables' else 'prevention' if file[
#                                                                                                                 :-4] == 'prevents' else 'intention' if file[
#                                                                                                                                                    :-4] == 'intends' else 'O'
#                 label='cause' if file[:-4] == 'causes' else 'enable' if file[
#                                                                             :-4] == 'enables' else 'prevent' if file[
#                                                                                                                 :-4] == 'prevents' else 'intend' if file[
#                                                                                                                                                    :-4] == 'intends' else '0'
#
#                 lab.append(label)
#                 tags = []
#                 pos_tags=[]
#                 postags=get_pos_data(nlp(row['sentence']))
#                 postags=list(filter(lambda i: i != 'SPACE', postags))
#
#
#                 for word,pos in zip(sent,postags):
#                     if word == event1:
#                         tags.append(tag1)
#                     elif word == event2:
#                         tags.append(tag2)
#                     else:
#                         tags.append(pos)
#                     pos_tags.append(pos)
#
#                 data.append(' '.join(tags))
#                 psd_tag_data.append(' '.join(pos_tags))
#
#                 # data.append([row['sentence'], label, file[:-4]])
#
# #
# #
#
#
# hong = read_file(hong_path[0])
# # #print(str(hong.groupby("rel1").count()))
# import re
#
#
# def get_hong_trigger(text):
#     r1 = re.compile("(.*?)\s*\(")
#     m1 = r1.match(text)
#     return m1.group(1)
#
# for index, row in hong.iterrows():
#     rel = get_relation(hong_path[0], row)
#     for r in rel:
#         if str(r).strip('2').strip().lower() not in ['causes', 'enables', 'intends', 'prevents']:
#             continue  # relation not known
#         event1 = row['evnt1text']
#         event2 = row['evnt2text']
#
#         trigger1=get_hong_trigger(event1)
#         trigger2=get_hong_trigger(event2)
#         tags = []
#         if '2' in str(r):  # swap the event order
#             temp = event1
#             event1 = event2
#             event2 = temp
#             temp = trigger1
#             trigger1 = trigger2
#             print(trigger1)
#             trigger2 = temp
#
#         rel = str(r).strip('2').strip().lower()
#
#         tag1 = 'cause' if rel == 'causes' else 'condition' if rel == 'enables' else 'O'
#         tag2 = 'effect'
#
#         # data.append(str(row['evnt1text']) + ' '+ str(row['evnt2text']))
#
#         sent = str(row['evnt1text']) + ' ' + str(row['evnt2text'])
#         text.append(sent)
#         pos_tag=get_pos_data(sent)
#         lbl='cause' if rel == 'causes' else 'enable' if rel == 'enables' else 'O'
#         lab.append(lbl)
#         sent = [token.text for token in nlp(sent)]
#         print(sent)
#
#
#
#         for word,tag in zip(sent,pos_tag):
#             # tags.append(tag)
#             if word == trigger1:
#                 tags.append(tag1)
#             elif word ==trigger2:
#                 tags.append(tag2)
#             else:
#                 tags.append(tag)
#         # for word in event2.split():
#         #     tags.append(tag2)
#
#         data.append(' '.join(tags))
# #
# read_nagative_data = pd.read_csv('data/out.csv', skiprows=506)
# read_nagative_data.columns=['id','sent','mean','rel']
# for index, row in read_nagative_data.iterrows():
#     lab.append('0')
#     tags = []
#     pos_tags=[]
#     text.append(row['sent'])
#     sent = [token.text for token in nlp(row['sent'])]
#     pstags = get_pos_data(nlp(row['sent']))
#     pstags = list(filter(lambda i: i != 'SPACE', pstags))
#     # sent=row['sent'].split()
#     for word,tag in zip(sent,pstags):
#         tags.append(tag)
#         pos_tags.append(tag)
#     data.append(' '.join(tags))
#     psd_tag_data.append(' '.join(pos_tags))

# print(read_nagative_data)
# data=[]
# new_enable=pd.read_csv('/Users/youssrarebboud/Desktop/enable_df.csv')
# print(new_enable)
#
# text=[]
# lab=[]
# for index, row in new_enable.iterrows():
#                 event1, event2 = row['trigger_1'],row['trigger_2']
#                 # data.append(row['sentence'])
#                 # sent = row['sentence'].split()
#                 text.append(row['sentence'])
#
#                 tag2 = 'effect'
#                 tag1 = 'condition'
#                 label='enable'
#                 lab.append(label)
#                 tags = []
#                 pos_tags=[]
#                 sent=[token.text for token in nlp(row['sentence'])]
#                 print(sent)
#
#
#
#                 for word in sent:
#                     print(event1)
#
#                     if word == event1:
#                         tags.append(tag1)
#                     elif word == event2:
#                         tags.append(tag2)
#                     else:
#                         tags.append('0')
#
#                 print(tags)
#                 data.append(' '.join(tags))
#
#
#                 # data.append([row['sentence'], label, file[:-4]])
#
# # #
# # #
# all = pd.DataFrame(data)
# # # all_pos=pd.DataFrame(pos_tag)
# text=pd.DataFrame(text)
# lbls=pd.DataFrame(lab)
# # # # texts=text['0']
# # # # pos_tags_pure=all_pos['0']
# # # # labels=all['0']
# # #
# # # # text.to_csv('data/txt.csv')
# # # # all.to_csv('data/out_df_events_tags.csv')  #
# # # # all_pos.to_csv('data/out_df_POStags.csv')
# # # # # with open("data/data_ER.txt", 'w') as f:
# # # # #     f.write("\n".join(map(str, data)))
# texts=text[text.columns[0]]
# NER=all[all.columns[0]]
# labels=lbls[lbls.columns[0]]
# # # lbls=lbls[lbls.columns[0]]
# new = pd.DataFrame()
# new['text']=texts
# new['tag']=NER
# # # new['NER']=labels
# new['label']=labels
# # #
# # print(new)
# # print(new['tag'])
# new.to_csv('new_enable_sp.csv')
# pure=pd.read_csv('pure.csv')
# prevent=pd.read_csv('new_prevent_sp.csv')
# enable=pd.read_csv('new_enable_sp.csv')
#
# new=pd.concat([pure, prevent])
# new=pd.concat([new,enable])
# print(new)
# new.to_csv('/Users/youssrarebboud/Desktop/new_pure.csv')
