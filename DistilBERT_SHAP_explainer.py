
import shap
import transformers
import pandas as pd
import torch
import numpy as np
import scipy as sp
from transformers import DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
device = torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
MODELNAME = None
model.load_state_dict(torch.load(MODELNAME))
model.to(device)

###

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

################################

# lab 1

def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).to(device)
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1])
    return val

#
explainer = shap.Explainer(f, tokenizer)

shap_values_lab1_200 = explainer(train_texts[0:200], fixed_context=1)
res = pd.DataFrame({'names':shap_values_lab1_200.sum(0).feature_names,'values':shap_values_lab1_200.sum(0).values})
res.to_excel('DistilBERT_shap_lab1_200_sign.xlsx')

# lab 2

def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).to(device)
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,2])
    return val

#
explainer = shap.Explainer(f, tokenizer)

shap_values_lab2_200 = explainer(train_texts[0:200], fixed_context=1)
res = pd.DataFrame({'names':shap_values_lab2_200.sum(0).feature_names,'values':shap_values_lab2_200.sum(0).values})
res.to_excel('DistilBERT_shap_lab2_200_sign.xlsx')

# lab 3

def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).to(device)
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,3])
    return val

#
explainer = shap.Explainer(f, tokenizer)

shap_values_lab3_200 = explainer(train_texts[0:200], fixed_context=1)
res = pd.DataFrame({'names':shap_values_lab3_200.sum(0).feature_names,'values':shap_values_lab3_200.sum(0).values})
res.to_excel('DistilBERT_shap_lab3_200_sign.xlsx')

