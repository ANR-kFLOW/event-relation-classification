import pandas as pd

enables = pd.read_csv(
    '/Users/youssrarebboud/PycharmProjects/GPT3DataAugmentation/generated_dataset/intention_clean.csv')

# enables=enables.drop_duplicates()
# print(enables)
# # print(enables.sentence.values)
# new=[]
# for idx, row in enables.iterrows():
#     new.append([re.sub('^[,\s]*','',row['sentence']),row['trigger1'],row['trigger2']])
#
# df = pd.DataFrame(new)
# df.columns = ['sentence', 'trigger1', 'trigger2']
# df=df.drop_duplicates()
# print(df)
# df.to_csv('/Users/youssrarebboud/PycharmProjects/GPT3DataAugmentation/generated_dataset/intends_manually_checked_dataset.csv')

# enable=enables.drop(enables[enables["sentence"].str.contains("to protect the rights")].index)
# enable.loc[enable['sentence'].str.contains("The government has implemented a series of laws to prevent the exploitation of natural resources."), 'trigger2'] = 'developing'
# enable.loc[enable['sentence'].str.contains("The government has implemented a series of laws to prevent the exploitation of natural resources."), 'trigger1'] = 'laws'
# enable.loc[enable['sentence'].str.contains("The government has implemented a series of laws to prevent the illegal trade of hazardous materials"), 'trigger2'] = 'illegal trade'
# enable.loc[enable['sentence'].str.contains("The government has implemented a series of laws to prevent the illegal trade of hazardous materials"), 'trigger1'] = 'laws'
# print(enable)

# enable.to_csv('/Users/youssrarebboud/PycharmProjects/GPT3DataAugmentation/generated_dataset/prevention1_manually_checked_dataset.csv')

correct_ones = []
correct_trigger1 = 0
correct_trigger2 = 0
correct_Sentenecs = 0

for idx, row in enables.iloc[:601].iterrows():

    print('line number :', idx)
    print(row['sentence'])
    print(row['trigger1'])
    print(row['trigger2'])
    inp = input('is the sentence correct?')
    if inp == '1':
        correct_Sentenecs += 1

        inp2 = input('is trigger1 correct?')
        inp3 = input('is trigger2 correct?')
        if inp2 == '1':
            correct_trigger1 += 1
            correct1 = row['trigger1']

        else:
            correct1 = input('correct please trigger1')

        if inp3 == '1':
            correct2 = row['trigger2']
            correct_trigger2 += 1
        else:
            correct2 = input('correct please trigger2')
        correct_ones.append([row['sentence'], correct1, correct2])
        df = pd.DataFrame(correct_ones)
        df.columns = ['sentence', 'trigger1', 'trigger2']
        # df.to_csv('/Users/youssrarebboud/PycharmProjects/GPT3DataAugmentation/generated_dataset/intention_manually_checked_dataset.csv')
        #

print('correct_Sentenecs: ', correct_Sentenecs)
print('correct_trigger1', correct_trigger1)
print('correct_trigger2', correct_trigger2)
