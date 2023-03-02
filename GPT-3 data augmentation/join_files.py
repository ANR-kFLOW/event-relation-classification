import pandas as pd

file1 = pd.read_csv('generated_event_triggers_GPT3_second_hit_intention.csv')

file2 = pd.read_csv('generated_event_triggers_GPT3_second_hit_intention_41.csv')
file3 = pd.read_csv('generated_event_triggers_GPT3_second_hit_intention_58.csv')

print(len(file2) + len(file3) + len(file1))

sentences = pd.read_csv('/Users/youssrarebboud/Desktop/intention_generated_gpt3_examples211.csv')
len(sentences)
result = pd.concat([file1, file2, file3])
print(len(result))
# result.columns=['0','events']
# new_row = {'0':'979', 'events':'(sanctions, invading)\n(Arms embargo, violence)\n(Sanctions,  nuclear)\n(arms embargo, abuse)'}
# result = result.append(new_row, ignore_index=True)
# # result=result.append(['0820','(sanctions, invading)\n\n(Arms embargo, violence)\n\n(Sanctions,  nuclear)\n\n(arms embargo, abuse)'])
# print(result)
#
result.to_csv('intention_gpt3.csv')
