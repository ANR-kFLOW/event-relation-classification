import warnings
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
from model import *

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# Set the device to use for computation
physical_devices = tf.config.list_physical_devices('GPU')
print('physical_devices', physical_devices)
tf.config.experimental.set_visible_devices(physical_devices, 'GPU')

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
BertTokenizer = bert_tokenization.FullTokenizer
warnings.filterwarnings("ignore")
#
# print(device_lib.list_local_devices())
# # Set the device to use for computation
# physical_devices=tf.config.list_physical_devices('GPU')
# print('physical_devices',physical_devices)
# tf.config.experimental.set_visible_devices(physical_devices, 'GPU')

# build_model()
BSA = build_model()
print(BSA.summary())

# data loading
ds = tf.data.Dataset.from_tensor_slices((X_train_token, y_train_label_NER, y_train_label_NER, label_train)).batch(16)
ds_val = tf.data.Dataset.from_tensor_slices((X_val_token, y_val_NER, y_val_NER, label_val)).batch(16)
ds_test = tf.data.Dataset.from_tensor_slices((X_test_token, y_test_label_NER, y_test_label_NER, label_test)).batch(16)

# define metrics
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


# define loss function for both events extraction and event relation extraction
def custum_crf_loss(potentials, sequence_length, chain_kernel, y):
    return tf.reduce_mean(-crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0])


loss_obj_cat = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

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

            crf_out, pred_out = BSA([X_train_token, y_train_label_NER])
            # print('pred out is   ',pred_out)
            # pred_out=pred_out[0, :]
            # print(pred_out)
            # print(y_pred)
            print('X_train_token')
            print(pred_out[0])

            viterbi_sequence, potentials, sequence_length, chain_kernel = crf_out
            print('viterbi_sequence')
            print(viterbi_sequence)
            print('sequence_length')
            print(sequence_length)
            print('potentials')
            print(potentials)
            print('chain_kernel')
            print(chain_kernel)
            print('y_train_label_NER')
            print(y_train_label_NER)
            # print(sequence_length.shape)
            print(label_train_NER.shape)
            crf_loss = custum_crf_loss(potentials, sequence_length, chain_kernel, y_train_label_NER)
            pred_loss = loss_obj_cat(label_train, pred_out)
            total_loss = crf_loss + pred_loss
            # y_pred = tf.argmax(pred_out, 1)

            # print(label_train.shape)
            # print(pred_out.shape)

        gradients = tape.gradient(total_loss, BSA.trainable_variables)
        optimizer.apply_gradients(zip(gradients, BSA.trainable_variables))
        loss_crf.update_state(crf_loss)
        loss_cat.update_state(pred_loss)
        # train_acc_metric_rel.update_state(label_train, pred_out)
        # crf_train_acc_metric.update_state(label_train_NER, viterbi_sequence,
        #                                   tf.sequence_mask(sequence_length, label_train_NER.shape[1]))
        # # Precision.update_state(labels, y_pred)
        # metric.update_state(label_train, pred_out)

        # for i,(X_train_token, y_train_label_NER, label_train_NER,label_train) in enumerate(ds):

        # X_train_token, y_train_label_NER, label_train = d

        print('done all the loop')

    # recal = {}
    # prec = {}
    # m = metric.result()
    # recall = np.diag(m) / np.sum(m, axis=1)
    # precision = np.diag(m) / np.sum(m, axis=0)
    print('epoch, {}'.format(epoch),
          'loss_type, {}'.format(loss_cat.result()),
          'crf_loss, {}'.format(loss_crf.result())
          # 'train_acc_rel, {}'.format(train_acc_metric_rel.result()),
          # 'crf train accurcy {}'.format(crf_train_acc_metric.result()),
          # 'metric relation {}'.format(metric.result()),
          # 'recall {}'.format(np.diag(m) / np.sum(m, axis=1)),
          # 'precision{}'.format(np.diag(m) / np.sum(m, axis=0))
          # 'crf per class {} '.format(per_class_accuracy_NER.result()),
          # 'label per class {}'.format((per_class_accuracy_rel.result()))

          # , 'crf_accuracy, {}'.format(train_metric_crf.result())
          )
    # for idx, item in enumerate(recall):
    #     recal[le.inverse_transform([idx])[0]] = item
    # for idx, item in enumerate(precision):
    #     prec[le.inverse_transform([idx])[0]] = item

    loss_crf.reset_states()
    loss_cat.reset_states()
    # train_acc_metric_rel.reset_states()
    # crf_train_acc_metric.reset_states()
    # metric.reset_states()
    # per_class_accuracy_NER.reset_states()
    # per_class_accuracy_rel.reset_states()
    # print(recal)
    # print(prec)

    logits_crf = []
    labels_crf = []
    logits_re = []
    labels_re = []
    logits_c = []
    gr_truth_crf = []

    for i, (X_val_token, label_val_NER, label_val_NER, label_val) in enumerate(ds_val):
        progbar.update(i)
        crf_out, pred_out = BSA([X_val_token, label_val_NER], training=False)
        viterbi_sequence, potentials, sequence_length, chain_kernel = crf_out
        logits_re.append(np.argmax(pred_out, axis=1).flatten())
        labels_re.append(label_val)
        # crf_layer = BSA.get_layer('crf_layer')
        # y_pred = crf_out[0]
        # sequence_lengths = crf_out[2]
        # viterbi_sequence, viterbi_score = crf_layer.viterbi_decode(y_pred, sequence_lengths)
        # print('viterbi_sequence',viterbi_sequence)
        for i in range(viterbi_sequence.shape[0]):
            label_clean = label_val_NER[i][label_val_NER[i] != 0]

            logits_clean = viterbi_sequence[i][label_val_NER[i] != 0]

            # print('len label_clean', len(label_clean))
            # print('len logits',len(logits_clean))
            # pre_report = logits_clean.detach().cpu().numpy()
            print('------logits_clean-------')
            print(logits_clean)
            # print('----------------')
            # print(label_clean)
            # print('-------------pre------------')
            #
            # predictions = logits_clean.argmax(dim=1)

            logits_crf.append(logits_clean)
            # print('----logits crf-----')
            # print(logits_crf)

            labels_crf.append(label_clean)
            # print('----lbl crf-----')
            # print(labels_crf)
            logits_c.append(list(itertools.chain(*logits_crf)))

    prediction_rp = list(itertools.chain(*logits_c))
    gt_crf = list(itertools.chain(*labels_crf))
    print('pprediction_rp')
    print(logits_c)
    result_tensor = tf.concat(logits_crf, axis=0)

    # Convert the result tensor to a numpy array to get a Python list
    result_list = result_tensor.numpy().tolist()
    print('result tensor')
    print(result_list)
    pred_re = list(itertools.chain(*logits_re))
    lbls_re = list(itertools.chain(*labels_re))
    # print('prediction itterated')
    # print(gt_re)
    report_crf = classification_report(gt_crf, result_list)

    print(report_crf)
    report_re = classification_report(lbls_re, pred_re)
    print(report_re)
BSA.save('joined_crf')
logits_crf = []
labels_crf = []
logits_c = []

BSA.save('joined_crf')
print('=======================testing starts ====================')
for i, (X_val_token, label_val_NER, label_val_NER, label_val) in enumerate(ds_test):
    progbar.update(i)
    crf_out, pred_out = BSA([X_val_token, label_val_NER], training=False)
    viterbi_sequence, potentials, sequence_length, chain_kernel = crf_out
    logits_re.append(np.argmax(pred_out, axis=1).flatten())
    labels_re.append(label_val)
    # crf_layer = BSA.get_layer('crf_layer')
    # y_pred = crf_out[0]
    # sequence_lengths = crf_out[2]
    # viterbi_sequence, viterbi_score = crf_layer.viterbi_decode(y_pred, sequence_lengths)
    # print('viterbi_sequence',viterbi_sequence)
    for i in range(viterbi_sequence.shape[0]):
        label_clean = label_val_NER[i][label_val_NER[i] != 0]

        logits_clean = viterbi_sequence[i][label_val_NER[i] != 0]

        # print('len label_clean', len(label_clean))
        # print('len logits',len(logits_clean))
        # pre_report = logits_clean.detach().cpu().numpy()
        print('------logits_clean-------')
        print(logits_clean)
        # print('----------------')
        # print(label_clean)
        # print('-------------pre------------')
        #
        # predictions = logits_clean.argmax(dim=1)

        logits_crf.append(logits_clean)
        # print('----logits crf-----')
        # print(logits_crf)

        labels_crf.append(label_clean)
        # print('----lbl crf-----')
        # print(labels_crf)
        logits_c.append(list(itertools.chain(*logits_crf)))

prediction_rp = list(itertools.chain(*logits_c))
gt_crf = list(itertools.chain(*labels_crf))
print('pprediction_rp')
print(logits_c)
result_tensor = tf.concat(logits_crf, axis=0)

# Convert the result tensor to a numpy array to get a Python list
result_list = result_tensor.numpy().tolist()
print('result tensor')
print(result_list)
pred_re = list(itertools.chain(*logits_re))
lbls_re = list(itertools.chain(*labels_re))
# print('prediction itterated')
# print(gt_re)
report_crf = classification_report(gt_crf, result_list)

print(report_crf)
report_re = classification_report(lbls_re, pred_re)
print(report_re)

BSA.save('joined_crf')
