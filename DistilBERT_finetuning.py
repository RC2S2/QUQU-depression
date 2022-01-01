
# DistilBERT for Depi text classif

# imports

import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from tqdm import tqdm, trange
from ast import literal_eval
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.metrics import cohen_kappa_score
from functools import reduce

# data
TEXTDIR = None
df7 = pd.read_pickle(TEXTDIR)
df7 = df7.dropna()

train, test = train_test_split(df7, test_size=0.25, random_state=12)

array = train.values
train_texts = array[:,28]
train_labels = array[:,2]
train_labels_2 = array[:,4]
array = test.values
test_texts = array[:,28]
test_labels = array[:,2]
test_labels_2 = array[:,4]


# tokenization

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

# dataset prep

import torch

class DepiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = DepiDataset(train_encodings, train_labels)
test_dataset = DepiDataset(test_encodings, test_labels)

# data loader

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
model.to(device)

model.train()

optim = AdamW(model.parameters(), lr=2e-5)

kappa_test = []

for epoch in range(3):
    model.train()
    print('epoch: '+ str(epoch))
    b_num = 0
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        print('batch number: ' + str(b_num))
        b_num = b_num+1
    #torch.save(model.state_dict(), 'dibert_b32_oc2_ep_' + str(epoch+1))
    model.eval()
    true_labels, pred_labels = [], []
    b_num = 0
    for batch in test_loader:
        with torch.no_grad():
            # Forward pass
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            pred_label = torch.sigmoid(outputs.logits)
            pred_label = pred_label.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
        print('batch number: ' + str(b_num))
        b_num = b_num + 1
        true_labels.append(labels)
        pred_labels.append(pred_label)
    # Flatten outputs
    true_labels = [item for sublist in true_labels for item in sublist]
    pred_test_labels = []
    for b in range(0, len(pred_labels)):
        b_pred_labels = pred_labels[b]
        for bb in range(0, len(b_pred_labels)):
            pred_test_labels.append(np.argmax(b_pred_labels[bb]))
    kappa_test.append(cohen_kappa_score(true_labels, pred_test_labels, labels=None, weights=None))
    print(kappa_test)

#

model.eval()

test_loader = DataLoader(test_dataset, batch_size=16)
true_labels, pred_labels = [], []
b_num = 0
for batch in test_loader:
    with torch.no_grad():
        # Forward pass
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        pred_label = torch.sigmoid(outputs.logits)
        pred_label = pred_label.to('cpu').numpy()
        labels = labels.to('cpu').numpy()
    print('batch number: ' + str(b_num))
    b_num = b_num+1
    true_labels.append(labels)
    pred_labels.append(pred_label)


true_labels = [item for sublist in true_labels for item in sublist]
pred_test_labels = []
for b in range(0,len(pred_labels)):
    b_pred_labels = pred_labels[b]
    for bb in range(0,len(b_pred_labels)):
        pred_test_labels.append(np.argmax(b_pred_labels[bb]))


#

y_test = true_labels
y_test_sec = test_labels_2
y_pred = pred_test_labels

#

kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
CR.append(classification_report(y_test, y_pred))
CM.append(confusion_matrix(y_test, y_pred))
kappa = cohen_kappa_score(y_test, y_pred, labels=None, weights=None)
kappa_list.append(kappa)
y_test_replaced = list(y_test)
for i in range(len(y_test)):
    if y_pred[i] != y_test[i] and y_pred[i] == y_test_sec[i]:
        y_test_replaced[i] = y_test_sec[i]
    else:
        continue

kappa_sec = cohen_kappa_score(y_test_replaced, y_pred, labels=None,weights=None)
kappa_sec_list.append(kappa_sec)
CR_sec.append(classification_report(y_test_replaced,y_pred))

def report_average(*args):
    report_list = list()
    for report in args:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        header = [x for x in splited[0].split(' ')]
        data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
        data = np.delete(data, 0, 1).astype(float)
        weighted_avg_total = np.array([x for x in splited[2].split(' ')][11:15]).astype(float).reshape(-1, len(header))
        df = pd.DataFrame(np.concatenate((data, weighted_avg_total)), columns=header)
        report_list.append(df)
    res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)
    return res.rename(index={res.index[-1]: 'avg / total'})


report_average_df = report_average(CR[0])
report_average_sec_df = report_average(CR_sec[0])

print(kappa_list)
print(kappa_sec_list)

print(report_average_df)
print(report_average_sec_df)


