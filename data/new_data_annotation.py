import pandas as pd
replacements = {'cause': 'trigger1', 'intention': 'trigger1', 'prevention': 'trigger1','condition':'trigger1'}

def modify_tags(df):
    df['tag_2'] = df['tag'].replace(replacements, regex=True)
    return df

train_original=pd.read_csv('original_train.csv')
val_origial=pd.read_csv('original_val.csv')
test_origial=pd.read_csv('original_test.csv')
train_joined=pd.read_csv('joined_train.csv')
val_joined=pd.read_csv('joined_val.csv')

modify_tags(train_original).to_csv('original_train.csv')
modify_tags(val_origial).to_csv('original_val.csv')
modify_tags(test_origial).to_csv('original_test.csv')
modify_tags(train_joined).to_csv('joined_train.csv')
modify_tags(val_joined).to_csv('joined_val.csv')

