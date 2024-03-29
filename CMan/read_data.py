import os

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing

max_len = 512
path_to_data = os.path.join('..', 'data')
train = pd.read_csv(path_to_data + '/joined_train.csv')
val = pd.read_csv(path_to_data + '/joined_val.csv')
test = pd.read_csv(path_to_data + '/original_test.csv')
print(train.columns)
# train=pd.read_csv('original_train.csv')
# val=pd.read_csv('original_val.csv')
# test=pd.read_csv('original_test.csv')
train['tags'] = train['tag_2'].str.replace('O', '0')
val['tags'] = val['tag_2'].str.replace('O', '0')
test['tags'] = test['tag_2'].str.replace('O', '0')

data = pd.concat([train, val], axis=0)

# data['tag'] = data['tag'].str.replace('0', 'N')
# X_train, X_val, y_train, y_val = train_test_split(data, data['label'], test_size=0.2, random_state=0, stratify=data['label'])
#
X_train_token = train['sentence']
y_train_label_NER = train['tags']

print(y_train_label_NER)

X_val_token = val['sentence']
y_val_NER = val['tags']

X_test_token = test['sentence']
y_test_label_NER = test['tags']

label_train = train['label']

label_val = val['label']

label_test = test['label']

vocab = ' '.join(data['tags'])

words = set(text_to_word_sequence(vocab))
print('words', words)
vocab_size = len(words)
print(vocab_size)

# Tokenize the tags in the "tag" column
# tokenizer = Tokenizer(num_words=vocab_size, lower=1, oov_token='<OOV>')

# words = [ 'n','intention', 'prevention', 'effect', 'condition', 'cause']

tokenizer = Tokenizer()
# tokenizer.word_index = {word: index for index, word in enumerate(words)}

tokenizer.fit_on_texts(data['tags'])
print(tokenizer.word_index)

# max_len = max([len(x) for x in tokenizer.texts_to_sequences(data['tags'])])
#
# print(max_len)
# Convert the tags to sequences and pad them
y_val_NER = tokenizer.texts_to_sequences(y_val_NER)
y_val_NER = pad_sequences(y_val_NER, maxlen=max_len, padding='post')

y_test_label_NER = tokenizer.texts_to_sequences(y_test_label_NER)
y_test_label_NER = pad_sequences(y_test_label_NER, maxlen=max_len, padding='post')

y_train_label_NER = tokenizer.texts_to_sequences(y_train_label_NER)
y_train_label_NER = pad_sequences(y_train_label_NER, maxlen=max_len, padding='post')
print(y_train_label_NER[1])
print(tokenizer.word_index)
# Print the resulting padded sequences

#
# tokenizer2 = Tokenizer(num_words=vocab_size, lower=1)
# tokenizer2.fit_on_texts(data['tag'].split())
# word_index2 = tokenizer2.word_index
# print(word_index2)
le = preprocessing.LabelEncoder()
le.fit(data['label'])
len(le.transform(data['label']))
# # print(le.transform(data['label']))
#
#
#
# y_train_label_NER= tokenizer2.texts_to_sequences(y_train_label_NER)
# y_val_NER = tokenizer2.texts_to_sequences(y_val_NER)
#
label_train = le.transform(label_train)
label_val = le.transform(label_val)
label_test = le.transform(label_test)
#
vocab = ' '.join(data['tags'])

words = set(text_to_word_sequence(vocab))
vocab_size = len(words)
print('vocan sizeof tags', vocab_size)
#

#
#
# y_train_label_NER = pad_sequences(y_train_label_NER, maxlen=max_len, padding='post', value=9)
# y_val_NER = pad_sequences(y_val_NER, maxlen=max_len, padding='post', value=9)
#
#
# # print(X_val_token.shape)
# #
# print(data['label'].value_counts())

# print("X_train_token:", X_train_token)
# print("y_train_label_NER:", y_train_label_NER)
# print("X_val_token:", X_val_token)
# print("y_val_NER:", list(y_val_NER))
# print("label_train:", label_train)
# print("label_val:", label_val)
# print(le.inverse_transform(label_val))
