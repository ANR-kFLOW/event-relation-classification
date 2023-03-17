from tensorflow_addons.text.crf import crf_log_likelihood
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
import itertools
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Concatenate, \
    MultiHeadAttention, Dropout
from tensorflow.keras import Model
import tensorflow_hub as hub
from read_data import *
import tensorflow as tf
from tf2crf import CRF
from bert import bert_tokenization




def build_model():

    # preprocessing input sentences and get bert word embeddings
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=True)
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=True)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    outputs = outputs['sequence_output']

    print('ok')
    BiLSTM = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name="BiLSTM")(
        outputs)
    print('here')
    att_layer = MultiHeadAttention(
        num_heads=4, key_dim=2, value_dim=2)
    output_tensor, weights = att_layer(BiLSTM, BiLSTM,
                                       return_attention_scores=True)
    print('done')
    merged = Concatenate(axis=1)([BiLSTM, output_tensor])
    Dense_layer_1 = Dense(100, activation='relu')(merged)

    Dropout_layer_1 = Dropout(0.5)(Dense_layer_1)
    BiLSTM = Bidirectional(LSTM(8, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name="BiLSTM23")(
        Dropout_layer_1)
    att_layer = MultiHeadAttention(
        num_heads=4, key_dim=2, value_dim=2)
    output_tensor, weights = att_layer(BiLSTM, BiLSTM,
                                       return_attention_scores=True)
    print('done')
    merged = Concatenate(axis=1)([BiLSTM, output_tensor])
    Dense_layer_1 = Dense(100, activation='relu')(merged)

    dense = Dense(vocab_size, activation="relu")(Dense_layer_1)

    crf = CRF(units=vocab_size + 1, dtype='float32', name='crf_layer')
    output = crf(dense)

    input_layer2 = Input(shape=(None,), name="Input_layer_BLA")

    embedding_layer = Embedding(input_dim=2000, output_dim=16,
                                # weights=[embedding_weights],
                                input_length=200,
                                trainable=False
                                )(input_layer2)
    BiLSTM = Bidirectional(LSTM(8, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name="BiLSTMBLA")(
        embedding_layer)
    # keys, values = tf.split(BiLSTM, num_or_size_splits=2, axis=1)
    att_layer = MultiHeadAttention(
        num_heads=4, key_dim=2, value_dim=2)
    output_tensor, weights = att_layer(query=outputs, value=BiLSTM, key=BiLSTM, return_attention_scores=True)

    merged = Concatenate(axis=1)([output_tensor, outputs])
    Dense_layer_1 = Dense(16, activation='relu')(merged)
    # merged_concat=keras.layers.Concatenate(axis=1)([Dense_layer_1, BSA])
    BiLSTM = Bidirectional(LSTM(8, return_sequences=True, recurrent_dropout=0.2, dropout=0.2),
                           name="BiLSTMBLAe")(
        Dense_layer_1)
    merged = Concatenate(axis=1)([BiLSTM, embedding_layer])
    BiLSTM = Bidirectional(LSTM(8, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name="BiLSTM2")(
        merged)
    att_layer = MultiHeadAttention(
        num_heads=4, key_dim=2, value_dim=2)

    output_tensor, weights = att_layer(query=BiLSTM, value=BiLSTM, return_attention_scores=True)

    merged = Concatenate(axis=1)([BiLSTM, output_tensor])
    BiLSTM = Bidirectional(LSTM(8, return_sequences=False, recurrent_dropout=0.2, dropout=0.2), name="BiLSTM24")(
        merged)
    Dense_layer_1 = Dense(100, activation='relu')(BiLSTM)
    flatten = tf.keras.layers.Flatten()(Dense_layer_1)

    Dropout_layer_1 = Dropout(0.5)(flatten)
    Dense_layer_2 = Dense(5, activation='softmax', name='second_output')(Dropout_layer_1)

    model = Model(
        inputs=[text_input, input_layer2],
        outputs=[output, Dense_layer_2], name="multioutput")
    # model_f=ModelWithCRFLoss(model, sparse_target=True)

    return model
