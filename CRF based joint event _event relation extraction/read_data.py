import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing
train=pd.read_csv('joined_train.csv')
val=pd.read_csv('joined_val.csv')
data = pd.concat([train, val], axis=0)

data['tag'] = data['tag'].str.replace('0', 'O')
X_train, X_val, y_train, y_val = train_test_split(data, data['label'], test_size=0.2, random_state=0, stratify=data['label'])
#
X_train_token = X_train['text']
y_train_label_NER = X_train['tag']


X_val_token = X_val['text']
y_val_NER = X_val['tag']


label_train = y_train


label_val=y_val


vocab = ' '.join(data['tag'])



words = set(text_to_word_sequence(vocab))
print('words', words)
vocab_size = len(words)
print(vocab_size)


tokenizer2 = Tokenizer(num_words=vocab_size, lower=1)
tokenizer2.fit_on_texts(data['tag'])
word_index2 = tokenizer2.word_index
print(word_index2)
le = preprocessing.LabelEncoder()
le.fit(data['label'])
len(le.transform(data['label']))


y_train_label_NER= tokenizer2.texts_to_sequences(y_train_label_NER)
y_val_NER = tokenizer2.texts_to_sequences(y_val_NER)

label_train=le.transform(label_train)
label_val=le.transform(label_val)


max_len=512


y_train_label_NER = pad_sequences(y_train_label_NER, maxlen=max_len, padding='post', value=9)
y_val_NER = pad_sequences(y_val_NER, maxlen=max_len, padding='post', value=9)


print(X_val_token.shape)

print(data['label'].value_counts())
