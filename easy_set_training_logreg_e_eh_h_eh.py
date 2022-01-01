############################
#  Features and texts, TFIDF vectorizer, char-ngram #
############################


import pandas as pd
import numpy as np
import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

os.chdir('C:/Users/NemethR/PycharmProjects/depression/')

of = open(os.path.join('models', 'logreg', 'easy_set_traning_logreg_e_eh_h_eh.tsv'), 'w',
          encoding='utf-8')

df7 = pd.read_pickle('c:/Users/NemethR/PycharmProjects/depression/data/interim/dataframes/df_pickled_final')
df7=df7.dropna()
# to choose easy-definition change variable name here
df7_copy = df7.copy()
df7_easy=df7_copy[df7['easy_1'] == 1].copy()
df7_noneasy=df7_copy[df7['easy_1'] == 999].copy()

classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000)

#separating easy and noneasy cases
array_e = df7_easy.values.copy()
x_txt_e = df7_easy.final_txt.values.astype('U').copy()
x_features_e = array_e[:,8:28].copy()
y_e = array_e[:,2].copy()
y_e = y_e.astype('int')
y2_e = array_e[:,4]
y2_e = y2_e.astype('int')

array_ne = df7_noneasy.values.copy()
x_txt_ne = df7_noneasy.final_txt.values.astype('U').copy()
x_features_ne = array_ne[:,8:28].copy()
y_ne = array_ne[:,2].copy()
y_ne = y_ne.astype('int')
y2_ne = array_ne[:,4]
y2_ne = y2_ne.astype('int')

# modell_e_eh: easy+hard test, easy training
# to change train/test roles, change here:
# x_txt_e, x_txt_ne, y2_ne, y_e, y_ne, x_features_ne, x_features_e
kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR2 = []
CM2 = []

x_train1_list = []
x_train2_list = []
x_train_list = []
x_test1_list = []
x_test2_list = []
x_test_list = []

y_test1_list = []
y2_test1_list = []
y_test2_list = []
y2_test2_list = []
y_test_list = []
y2_test_list = []
y_train_list = []
y_train1_list = []
y_train2_list = []
x_features_train_list = []
x_features_train1_list = []
x_features_train2_list = []
x_features_test1_list = []
x_features_test2_list = []
x_features_test_list = []

# hard set division 4/5,1/5 for train and test (1 means hard)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
for train_index, test_index in tqdm.tqdm(kf.split(x_txt_ne, y_ne)):
    x_train1 = x_txt_ne[train_index]
    x_train1_list.append(x_train1)
    y_train1 = y_ne[train_index]
    y_train1_list.append(y_train1)
    x_features_train1 = x_features_ne[train_index]
    x_features_train1_list.append(x_features_train1)
    x_test1 = x_txt_ne[test_index]
    x_test1_list.append(x_test1)
    y_test1 = y_ne[test_index]
    y_test1_list.append(y_test1)
    y2_test1 = y2_ne[test_index]
    y2_test1_list.append(y2_test1)
    x_features_test1 = x_features_ne[test_index]
    x_features_test1_list.append(x_features_test1)

# easy set division 4/5,1/5 for train and test (2 means easy)
for train_index, test_index in tqdm.tqdm(kf.split(x_txt_e, y_e)):
    x_test2 = x_txt_e[test_index]
    x_test2_list.append(x_test2)
    y_test2 = y_e[test_index]
    y_test2_list.append(y_test2)
    y2_test2 = y2_e[test_index]
    y2_test2_list.append(y2_test2)
    x_features_test2 = x_features_e[test_index]
    x_features_test2_list.append(x_features_test2)
    x_train2 = x_txt_e[train_index]
    x_train2_list.append(x_train2)
    y_train2 = y_e[train_index]
    y_train2_list.append(y_train2)
    x_features_train2 = x_features_e[train_index]
    x_features_train2_list.append(x_features_train2)

# create test set fom 1/5 easy and 1/5 hard
for x in range(0, 5):
    x_test_list.append(np.concatenate((x_test1_list[x], x_test2_list[x]), axis=0))
    y_test_list.append(np.concatenate((y_test1_list[x], y_test2_list[x]), axis=0))
    y2_test_list.append(np.concatenate((y2_test1_list[x], y2_test2_list[x]), axis=0))
    x_features_test_list.append(np.concatenate((x_features_test1_list[x], x_features_test2_list[x]), axis=0))

for x in range(0, 5):
    vector = TfidfVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(x_train2_list[x])
    # easy 4/5 lesz train
    x_train2_vect = vector.transform(x_train2_list[x])
    # fent létrejött kapcsolt lesz teszt
    x_test_vect = vector.transform(x_test_list[x])
    x_train2_fts = x_features_train2_list[x]
    x_test_fts = x_features_test_list[x]
    X_train = np.column_stack((x_train2_vect.A, x_train2_fts))
    X_test = np.column_stack((x_test_vect.A, x_test_fts))
    y_train = y_train2_list[x]
    y_test = y_test_list[x]
    y_test_sec = y2_test_list[x]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
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
    kappa_sec = cohen_kappa_score(y_test_replaced, y_pred, labels=None,
                                  weights=None)
    kappa_sec_list.append(kappa_sec)
    CR2.append(classification_report(y_test_replaced, y_pred))
    CM2.append(confusion_matrix(y_test_replaced, y_pred))
print(kappa_list, kappa_sec_list)

of.write("modell_e_eh: easy trainig, easy+hard test" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
for l in CR:
    of.write(l)
    of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("classification report (modified version):" + "\n")
for l in CR2:
    of.write(l)
    of.write("\n")
of.write("\n")
of.write("confusion matrix (modified version):" + "\n")
for l in CM2:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")

print("modell_e_he done")

# modell_e_eh: easy+hard test, hard training
# to change train/test roles, change here:
# x_txt_e, x_txt_ne, y2_ne, y_e, y_ne, x_features_ne, x_features_e
kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR2 = []
CM2 = []

x_train1_list = []
x_train2_list = []
x_train_list = []
x_test1_list = []
x_test2_list = []
x_test_list = []

y_test1_list = []
y2_test1_list = []
y_test2_list = []
y2_test2_list = []
y_test_list = []
y2_test_list = []
y_train_list = []
y_train1_list = []
y_train2_list = []
x_features_train_list = []
x_features_train1_list = []
x_features_train2_list = []
x_features_test1_list = []
x_features_test2_list = []
x_features_test_list = []

# hard set division 4/5,1/5 for train and test (1 means hard)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
for train_index, test_index in tqdm.tqdm(kf.split(x_txt_ne, y_ne)):
    x_train1 = x_txt_ne[train_index]
    x_train1_list.append(x_train1)
    y_train1 = y_ne[train_index]
    y_train1_list.append(y_train1)
    x_features_train1 = x_features_ne[train_index]
    x_features_train1_list.append(x_features_train1)
    x_test1 = x_txt_ne[test_index]
    x_test1_list.append(x_test1)
    y_test1 = y_ne[test_index]
    y_test1_list.append(y_test1)
    y2_test1 = y2_ne[test_index]
    y2_test1_list.append(y2_test1)
    x_features_test1 = x_features_ne[test_index]
    x_features_test1_list.append(x_features_test1)

# easy set division 4/5,1/5 for train and test (2 means easy)
for train_index, test_index in tqdm.tqdm(kf.split(x_txt_e, y_e)):
    x_test2 = x_txt_e[test_index]
    x_test2_list.append(x_test2)
    y_test2 = y_e[test_index]
    y_test2_list.append(y_test2)
    y2_test2 = y2_e[test_index]
    y2_test2_list.append(y2_test2)
    x_features_test2 = x_features_e[test_index]
    x_features_test2_list.append(x_features_test2)
    x_train2 = x_txt_e[train_index]
    x_train2_list.append(x_train2)
    y_train2 = y_e[train_index]
    y_train2_list.append(y_train2)
    x_features_train2 = x_features_e[train_index]
    x_features_train2_list.append(x_features_train2)

# create test set fom 1/5 easy and 1/5 hard
for x in range(0, 5):
    x_test_list.append(np.concatenate((x_test1_list[x], x_test2_list[x]), axis=0))
    y_test_list.append(np.concatenate((y_test1_list[x], y_test2_list[x]), axis=0))
    y2_test_list.append(np.concatenate((y2_test1_list[x], y2_test2_list[x]), axis=0))
    x_features_test_list.append(np.concatenate((x_features_test1_list[x], x_features_test2_list[x]), axis=0))

for x in range(0, 5):
    vector = TfidfVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(x_train1_list[x])
    x_train1_vect = vector.transform(x_train1_list[x])
    x_test_vect = vector.transform(x_test_list[x])
    x_train1_fts = x_features_train1_list[x]
    x_test_fts = x_features_test_list[x]
    X_train = np.column_stack((x_train1_vect.A, x_train1_fts))
    X_test = np.column_stack((x_test_vect.A, x_test_fts))
    y_train = y_train1_list[x]
    y_test = y_test_list[x]
    y_test_sec = y2_test_list[x]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
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
    kappa_sec = cohen_kappa_score(y_test_replaced, y_pred, labels=None,
                                  weights=None)
    kappa_sec_list.append(kappa_sec)
    CR2.append(classification_report(y_test_replaced, y_pred))
    CM2.append(confusion_matrix(y_test_replaced, y_pred))
print(kappa_list, kappa_sec_list)

of.write("modell_e_eh: hard trainig, easy+hard test" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
for l in CR:
    of.write(l)
    of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("classification report (modified version):" + "\n")
for l in CR2:
    of.write(l)
    of.write("\n")
of.write("\n")
of.write("confusion matrix (modified version):" + "\n")
for l in CM2:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")

print("modell_h_he done")

of.close()
