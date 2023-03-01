import warnings
from keras.preprocessing.text import text_to_word_sequence
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tensorflow_addons.text.crf import crf_log_likelihood
warnings.filterwarnings("ignore")
import spacy
from sklearn.metrics import classification_report
import itertools
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Concatenate, \
    MultiHeadAttention, Dropout, Lambda

from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_hub as hub

from read_data import *
import tensorflow as tf
from tf2crf import CRF
from bert import bert_tokenization
import os
import tensorflow_addons as tfa

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
BertTokenizer = bert_tokenization.FullTokenizer

# from bert.tokenization import FullTokenizer
import tensorflow_text as text




bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=True)
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)

vocab = ' '.join(data['tag'])



words = set(text_to_word_sequence(vocab))
vocab_size = len(words)



def build_model():
    # preprocessing input sentences and get bert word embeddings

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
    Dense_layer_1 = Dense(8336, activation='relu')(merged)

    Dropout_layer_1 = Dropout(0.5)(Dense_layer_1)
    BiLSTM = Bidirectional(LSTM(8, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name="BiLSTM23")(
        Dropout_layer_1)
    att_layer = MultiHeadAttention(
        num_heads=4, key_dim=2, value_dim=2)
    output_tensor, weights = att_layer(BiLSTM, BiLSTM,
                                       return_attention_scores=True)
    print('done')
    merged = Concatenate(axis=1)([BiLSTM, output_tensor])
    Dense_layer_1 = Dense(8336, activation='relu')(merged)

    dense = Dense(vocab_size, activation="relu")(Dense_layer_1)

    crf = CRF(units=vocab_size, dtype='float32', name='crf_layer')
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
    Dense_layer_1 = Dense(8336, activation='relu')(BiLSTM)
    flatten = tf.keras.layers.Flatten()(Dense_layer_1)

    Dropout_layer_1 = Dropout(0.5)(flatten)
    Dense_layer_2 = Dense(5, activation='softmax', name='second_output')(Dropout_layer_1)

    model = Model(
        inputs=[text_input, input_layer2],
        outputs=[output, Dense_layer_2], name="multioutput")
    # model_f=ModelWithCRFLoss(model, sparse_target=True)

    return model


# build_model()
BSA = build_model()
print(BSA.summary())

# data loading

ds = tf.data.Dataset.from_tensor_slices((X_train_token, y_train_label_NER, y_train_label_NER, label_train)).batch(16)
ds_val = tf.data.Dataset.from_tensor_slices((X_val_token, y_val_NER, y_val_NER, label_val)).batch(16)

#define metrics
class MultiClassConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="multi_class_confusion_matrix", **kwargs):
        super(MultiClassConfusionMatrix, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes

        self.mctp_conf_matrix = self.add_weight(name="confusion_matrix", shape=(num_classes, num_classes),
                                                dtype=tf.dtypes.int32, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(y_pred, axis=-1)

        tmp = tf.math.confusion_matrix(y_true, y_pred, self.num_classes)
        res = tf.math.add(tmp, self.mctp_conf_matrix)

        self.mctp_conf_matrix.assign(res)

    def result(self):
        return self.mctp_conf_matrix

    def reset_states(self):
        """
        In version 2.5.0 this method is renamed to "reset_state"
        """
        self.mctp_conf_matrix.assign(tf.zeros((self.num_classes, self.num_classes), dtype=tf.dtypes.int32))

#define loss function for both events extraction and event relation extraction
def custum_crf_loss(potentials, sequence_length, chain_kernel, y):
    return tf.reduce_mean(-crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0])


loss_obj_cat = tf.keras.losses.SparseCategoricalCrossentropy()



optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# loss_reg = tf.keras.metrics.Mean(name='regression loss')
# loss_cat = tf.keras.metrics.Mean(name='categorical loss')
loss_crf = tf.keras.metrics.Mean(name='crf loss')
loss_cat = tf.keras.metrics.Mean(name='categorical loss')

# error_cat = tf.keras.metrics.SparseCategoricalAccuracy()
# train_metric_crf = tf.keras.metrics.Accuracy(name='accuracy')
train_acc_metric_rel = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
val_acc_metric_rel = tf.keras.metrics.SparseCategoricalAccuracy(name='accu')
sss = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
crf_train_acc_metric = tf.keras.metrics.Accuracy(name='accuracy')
crf_val_acc_metric = tf.keras.metrics.Accuracy(name='accuracyval')
Precision = tf.keras.metrics.Precision()
metric = MultiClassConfusionMatrix(num_classes=5)
metric_val = MultiClassConfusionMatrix(num_classes=5)

progbar = tf.keras.utils.Progbar(len(ds))
for epoch in range(50):
    print("\nStart of epoch %d" % (epoch,))

    for i, (X_train_token, y_train_label_NER, label_train_NER, label_train) in enumerate(ds):
        progbar.update(i)
        with tf.GradientTape() as tape:
            # input_crf,input_rel=inputsi
            print('I am here _')
            crf_out, pred_out = BSA([X_train_token, y_train_label_NER])
            # print('pred out is   ',pred_out)
            # pred_out=pred_out[0, :]
            # print(pred_out)
            # print(y_pred)

            viterbi_sequence, potentials, sequence_length, chain_kernel = crf_out

            # print(potentials.shape)
            # print(sequence_length.shape)
            crf_loss = custum_crf_loss(potentials, sequence_length, chain_kernel, viterbi_sequence)
            pred_loss = loss_obj_cat(label_train, pred_out)
            total_loss = crf_loss + pred_loss
            # y_pred = tf.argmax(pred_out, 1)

            # print(label_train.shape)
            # print(pred_out.shape)

        gradients = tape.gradient(total_loss, BSA.trainable_variables)
        optimizer.apply_gradients(zip(gradients, BSA.trainable_variables))
        loss_crf.update_state(crf_loss)
        loss_cat.update_state(pred_loss)
        train_acc_metric_rel.update_state(label_train, pred_out)
        crf_train_acc_metric.update_state(label_train_NER, viterbi_sequence,
                                          tf.sequence_mask(sequence_length, label_train_NER.shape[1]))
        # Precision.update_state(labels, y_pred)
        metric.update_state(label_train, pred_out)

        # for i,(X_train_token, y_train_label_NER, label_train_NER,label_train) in enumerate(ds):

        # X_train_token, y_train_label_NER, label_train = d

        print('done all the loop')

    recal = {}
    prec = {}
    m = metric.result()
    recall = np.diag(m) / np.sum(m, axis=1)
    precision = np.diag(m) / np.sum(m, axis=0)
    print('epoch, {}'.format(epoch),
          'loss_type, {}'.format(loss_cat.result()),
          'crf_loss, {}'.format(loss_crf.result()),
          'train_acc_rel, {}'.format(train_acc_metric_rel.result()),
          'crf train accurcy {}'.format(crf_train_acc_metric.result()),
          'metric relation {}'.format(metric.result()),
          'recall {}'.format(np.diag(m) / np.sum(m, axis=1)),
          'precision{}'.format(np.diag(m) / np.sum(m, axis=0))
          # 'crf per class {} '.format(per_class_accuracy_NER.result()),
          # 'label per class {}'.format((per_class_accuracy_rel.result()))

          # , 'crf_accuracy, {}'.format(train_metric_crf.result())
          )
    for idx, item in enumerate(recall):
        recal[le.inverse_transform([idx])[0]] = item
    for idx, item in enumerate(precision):
        prec[le.inverse_transform([idx])[0]] = item

    loss_crf.reset_states()
    loss_cat.reset_states()
    train_acc_metric_rel.reset_states()
    crf_train_acc_metric.reset_states()
    metric.reset_states()
    # per_class_accuracy_NER.reset_states()
    # per_class_accuracy_rel.reset_states()
    print(recal)
    print(prec)

    logits_crf = []
    labels_crf = []
    logits_re = []
    labels_re = []

    for i, (X_val_token, label_val_NER, label_val_NER, label_val) in enumerate(ds_val):
        progbar.update(i)
        crf_out, pred_out = BSA([X_val_token, label_val_NER], training=False)
        viterbi_sequence, potentials, sequence_length, chain_kernel = crf_out
        logits_re.append(np.argmax(pred_out, axis=1).flatten())
        labels_re.append(label_val)
        crf_layer = BSA.get_layer('crf_layer')
        y_pred = crf_out[0]
        sequence_lengths = crf_out[2]
        # viterbi_sequence, viterbi_score = crf_layer.viterbi_decode(y_pred, sequence_lengths)
        # print('viterbi_sequence',viterbi_sequence)
        for i in range(viterbi_sequence.shape[0]):
            label_clean = label_val_NER[i][label_val_NER[i] != 9]

            logits_clean = viterbi_sequence[i][label_val_NER[i] != 9]

            # print('len label_clean', len(label_clean))
            # print('len logits',len(logits_clean))
            # pre_report = logits_clean.detach().cpu().numpy()
            # print('----------------')
            # print(logits_clean)
            # print('----------------')
            # print(label_clean)
            # print('-------------pre------------')
            #
            # predictions = logits_clean.argmax(dim=1)

            logits_crf.append(logits_clean)
            # print(logits_crf)

            labels_crf.append(label_clean)
            # print('------------')
            # print(labels_crf)

        prediction_rp = list(itertools.chain(*logits_crf))
        gt_re = list(itertools.chain(*labels_crf))
        pred_re = list(itertools.chain(*logits_re))
        lbls_re = list(itertools.chain(*labels_re))
        # print('prediction itterated')
        # print(gt_re)
        report_crf = classification_report(gt_re, prediction_rp)

        print(report_crf)
        report_re = classification_report(lbls_re, pred_re)
        print(report_re)

        # Update val metrics
#         val_acc_metric_rel.update_state(label_val, pred_out)
#         crf_val_acc_metric.update_state(label_val_NER, viterbi_sequence, tf.sequence_mask(sequence_length, label_val_NER.shape[1]))
#         metric_val.update_state(label_val, pred_out)
# #
#     recal_val={}
#     prec_val={}
#     val_acc = val_acc_metric_rel.result()
#     crf_val_acc=crf_val_acc_metric.result()
#     metric_val_re=metric_val.result()
#     recall_val = np.diag(metric_val_re) / np.sum(metric_val_re, axis=1)
#     precision_val = np.diag(metric_val_re) / np.sum(metric_val_re, axis=0)
#     for idx, item in enumerate(recall_val):
#         recal_val[le.inverse_transform([idx])[0]] = item
#     for idx, item in enumerate(precision_val):
#         prec_val[le.inverse_transform([idx])[0]] = item
#     val_acc_metric_rel.reset_states()
#     crf_val_acc_metric.reset_states()
#     metric_val.reset_states()
#
# #
# #
# #
#     print("Validation acc: %.4f" % (float(val_acc),))
#     print("crf Validation acc: %.4f" % (float(crf_val_acc),))
#     print("precision: %.4f" )
#     print(prec_val)
#     print("recall: %.4f")
#     print(recal_val)


# BSA.save('model_bert_Pos_tag')
