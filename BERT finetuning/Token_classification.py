from os import path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast

ROOT = 'data/'

label_all_tokens = False
# read data, this setting is for training and testing on original data, change the data file to
# joined_train and joined_val to test on the new dataset
df_train = pd.read_csv(path.join(ROOT, 'joined_train.csv'))
df_val = pd.read_csv(path.join(ROOT, 'joined_val.csv'))
df_test = pd.read_csv(path.join(ROOT, 'original_test.csv'))
labels = [word_tokenize(i) for i in df_train['tag'].values.tolist()]

# Check how many labels are there in the dataset
unique_labels = set()

for lb in labels:
    [unique_labels.add(i) for i in lb if i not in unique_labels]

print(unique_labels)

# Map each label into its id representation and vice versa
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
print(labels_to_ids)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):
        lb = [word_tokenize(i) for i in df['tag'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                                padding='max_length', max_length=512, truncation=True, return_tensors="pt") for i in
                      txt]
        self.labels = [align_label(i, j) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


def masked_loss(logits, targets, ignore_index):
    # Create a mask for the ignored classes

    mask = torch.zeros_like(targets)
    for idx in ignore_index:
        mask = mask + (targets == idx)
    mask = 1 - mask.byte()

    # Compute the cross-entropy loss with the mask applied
    loss = F.cross_entropy(logits, targets, reduction='none')
    masked_loss = loss * mask.float()
    masked_loss = masked_loss.sum() / mask.float().sum()
    return masked_loss


class BertModel(torch.nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, tag):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=tag, return_dict=False)
        # output2=

        return output


a = []


def train_loop(model, df_train, df_val):
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # create a scheduler that reduces the learning rate by a factor of 0.1 every 10 epochs
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)
            # loop over each sample in the  batch
            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        pred = []
        gt = []

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)
            # loop over each sample of the batch and get the predicted tokens and their appropriate tags, save them in a list to be able to check the performance for each token class

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]
                pre_report = logits_clean.detach().cpu().numpy()
                print('----------------')
                print(logits.shape[0])
                print(logits.shape)
                print('----------------')
                print(logits_clean)
                print('----------------')
                print(label_clean)
                print('-------------pre------------')

                predictions = logits_clean.argmax(dim=1)

                pred.append(np.argmax(pre_report, axis=1).flatten())
                print(pred)

                gt.append(label_clean.to('cpu').numpy())
                print('------------')
                print(gt)

                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        # adjust the learning rate using the scheduler
        scheduler.step()
        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)
        prediction_rp = list(itertools.chain(*pred))
        gt_re = list(itertools.chain(*gt))
        print('prediction itterated')
        print(gt_re)
        report = classification_report(gt_re, prediction_rp)
        print(report)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .6f} | Accuracy: {total_acc_train / len(df_train): .6f} | Val_Loss: {total_loss_val / len(df_val): .6f} | Accuracy: {total_acc_val / len(df_val): .6f}')


LEARNING_RATE = 5e-3
EPOCHS = 10
BATCH_SIZE = 8

model = BertModel()
train_loop(model, df_train, df_val)

import itertools


# testing the model on the test set
def evaluate(model, df_test):
    pred = []
    gt = []
    test_dataset = DataSequence(df_test)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

        test_label = test_label.to(device)
        mask = test_data['attention_mask'].squeeze(1).to(device)

        input_id = test_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, test_label)

        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            pre_report = logits_clean.detach().cpu().numpy()

            pred.append(np.argmax(pre_report, axis=1).flatten())

            gt.append(label_clean.to('cpu').numpy())

            # lbls.append(label_clean)
            acc = (predictions == label_clean).float().mean()
            total_acc_test += acc

    prediction_rp = list(itertools.chain(*pred))
    gt_re = list(itertools.chain(*gt))
    report = classification_report(gt_re, prediction_rp)
    print(report)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .6f}')
    print(report)


evaluate(model, df_test)

PATH = "entire_model_token_classification_original_data.pt"

# Save the model
# torch.save(model, PATH)
