import pandas as pd
import numpy as np
import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from functools import reduce

os.chdir('C:/Users/NemethR/PycharmProjects/depression/')

df7 = pd.read_pickle('data/interim/dataframes/df_pickled_final')
df7 = df7.dropna()
array = df7.values

classifier = LogisticRegression(multi_class='auto', solver='newton-cg', max_iter=10000)

x_txt = df7.final_txt.values.astype('U')
x_features = array[:,8:28]

y = array[:,2]
y = y.astype('int')
y2 = array[:,4]
y2 = y2.astype('int')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)


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


of = open('models/logreg/final_oc2_logreg.tsv','w',encoding='utf-8')


############################
#       K-FOLD LOGREG      #
#   Texts: Tfidf + Count   #
############################


# COUNT VECTORIZER


kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
for train_index, test_index in tqdm.tqdm(kf.split(x_txt,y)):
    x_train, x_test = x_txt[train_index], x_txt[test_index]
    vector = CountVectorizer().fit(x_train)
    X_train = vector.transform(x_train)
    X_test = vector.transform(x_test)
    y_train, y_test = y[train_index], y[test_index]
    y_test_sec = y2[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    CR.append(classification_report(y_test,y_pred))
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
    CR_sec.append(classification_report(y_test_replaced,y_pred))


report_average_sec_df = report_average(CR_sec[0], CR_sec[1], CR_sec[2], CR_sec[3], CR_sec[4])
report_average_df = report_average(CR[0], CR[1], CR[2], CR[3], CR[4])
of.write("COUNTVECT ON TXT" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
of.write(report_average_df.to_string())
of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")
of.write("classification report sec:" + "\n")
of.write(report_average_sec_df.to_string())
of.write("\n")
of.write("\n")

print("COUNTVECT ON TXT is done")


# TFIDF VECTORIZER


kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
for train_index, test_index in tqdm.tqdm(kf.split(x_txt,y)):
    x_train, x_test = x_txt[train_index], x_txt[test_index]
    vector = TfidfVectorizer(min_df=5).fit(x_train)
    X_train = vector.transform(x_train)
    X_test = vector.transform(x_test)
    y_train, y_test = y[train_index], y[test_index]
    y_test_sec = y2[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    CR.append(classification_report(y_test,y_pred))
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
    CR_sec.append(classification_report(y_test_replaced,y_pred))

report_average_sec_df = report_average(CR_sec[0], CR_sec[1], CR_sec[2], CR_sec[3], CR_sec[4])
report_average_df = report_average(CR[0], CR[1], CR[2], CR[3], CR[4])

of.write("TFIDFVECT ON TXT" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
of.write(report_average_df.to_string())
of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")
of.write("classification report sec:" + "\n")
of.write(report_average_sec_df.to_string())
of.write("\n")
of.write("\n")

print("TFIDFVECT ON TXT is done")


############################
#       K-FOLD LOGREG      #
#    Features and texts    #
############################


# COUNT VECTORIZER


kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
for train_index, test_index in tqdm.tqdm(kf.split(x_txt,y)):
    x_train, x_test = x_txt[train_index], x_txt[test_index]
    vector = CountVectorizer().fit(x_train)
    x_train_vect = vector.transform(x_train)
    x_test_vect = vector.transform(x_test)
    x_train_fts = x_features[train_index]
    x_test_fts = x_features[test_index]
    X_train = np.column_stack((x_train_vect.A, x_train_fts))
    X_test = np.column_stack((x_test_vect.A, x_test_fts))
    y_train, y_test = y[train_index], y[test_index]
    y_test_sec = y2[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    CR.append(classification_report(y_test,y_pred))
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
    CR_sec.append(classification_report(y_test_replaced,y_pred))

report_average_sec_df = report_average(CR_sec[0], CR_sec[1], CR_sec[2], CR_sec[3], CR_sec[4])
report_average_df = report_average(CR[0], CR[1], CR[2], CR[3], CR[4])
of.write("COUNTVECT ON TXT AND FEATURES" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
of.write(report_average_df.to_string())
of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")
of.write("classification report sec:" + "\n")
of.write(report_average_sec_df.to_string())
of.write("\n")
of.write("\n")

print("COUNTVECT ON TXT AND FEATURES is done")


# TFIDF VECTORIZER


kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
for train_index, test_index in tqdm.tqdm(kf.split(x_txt,y)):
    x_train, x_test = x_txt[train_index], x_txt[test_index]
    vector = TfidfVectorizer(min_df=5).fit(x_train)
    x_train_vect = vector.transform(x_train)
    x_test_vect = vector.transform(x_test)
    x_train_fts = x_features[train_index]
    x_test_fts = x_features[test_index]
    X_train = np.column_stack((x_train_vect.A, x_train_fts))
    X_test = np.column_stack((x_test_vect.A, x_test_fts))
    y_train, y_test = y[train_index], y[test_index]
    y_test_sec = y2[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    CR.append(classification_report(y_test,y_pred))
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
    CR_sec.append(classification_report(y_test_replaced,y_pred))

report_average_sec_df = report_average(CR_sec[0], CR_sec[1], CR_sec[2], CR_sec[3], CR_sec[4])
report_average_df = report_average(CR[0], CR[1], CR[2], CR[3], CR[4])
of.write("TFIDFVECT ON TXT AND FEATURES" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
of.write(report_average_df.to_string())
of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")
of.write("classification report sec:" + "\n")
of.write(report_average_sec_df.to_string())
of.write("\n")
of.write("\n")

print("TFIDFTVECT ON TXT AND FEATURES is done")


####################################
#    COMBINE FEATURES AND TEXT     #
#         TFIDF VECTORIZER         #
#      LOGREG ON WORD N-GRAMS      #
####################################


kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
for train_index, test_index in tqdm.tqdm(kf.split(x_txt,y)):
    x_train, x_test = x_txt[train_index], x_txt[test_index]
    vector = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(x_train)
    x_train_vect = vector.transform(x_train)
    x_test_vect = vector.transform(x_test)
    x_train_fts = x_features[train_index]
    x_test_fts = x_features[test_index]
    X_train = np.column_stack((x_train_vect.A, x_train_fts))
    X_test = np.column_stack((x_test_vect.A, x_test_fts))
    y_train, y_test = y[train_index], y[test_index]
    y_test_sec = y2[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    CR.append(classification_report(y_test,y_pred))
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
    CR_sec.append(classification_report(y_test_replaced,y_pred))

report_average_sec_df = report_average(CR_sec[0], CR_sec[1], CR_sec[2], CR_sec[3], CR_sec[4])
report_average_df = report_average(CR[0], CR[1], CR[2], CR[3], CR[4])

of.write("TFIDFVECT ON TXT AND FEATURES WORDNGRAMS" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
of.write(report_average_df.to_string())
of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")
of.write("classification report sec:" + "\n")
of.write(report_average_sec_df.to_string())
of.write("\n")
of.write("\n")

print("TFIDFTVECT ON TXT AND FEATURES WORDNGRAMS is done")


####################################
#    COMBINE FEATURES AND TEXT     #
#         TFIDF VECTORIZER         #
#      LOGREG ON CHAR N-GRAMS      #
####################################


kappa_list = []
kappa_sec_list = []
CR = []
CM = []
CR_sec = []
for train_index, test_index in tqdm.tqdm(kf.split(x_txt,y)):
    x_train, x_test = x_txt[train_index], x_txt[test_index]
    vector = TfidfVectorizer(min_df=5, ngram_range=(2,5),
                             analyzer='char_wb').fit(x_train)
    x_train_vect = vector.transform(x_train)
    x_test_vect = vector.transform(x_test)
    x_train_fts = x_features[train_index]
    x_test_fts = x_features[test_index]
    X_train = np.column_stack((x_train_vect.A, x_train_fts))
    X_test = np.column_stack((x_test_vect.A, x_test_fts))
    y_train, y_test = y[train_index], y[test_index]
    y_test_sec = y2[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    CR.append(classification_report(y_test,y_pred))
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
    CR_sec.append(classification_report(y_test_replaced,y_pred))

report_average_sec_df = report_average(CR_sec[0], CR_sec[1], CR_sec[2], CR_sec[3], CR_sec[4])
report_average_df = report_average(CR[0], CR[1], CR[2], CR[3], CR[4])

of.write("TFIDFVECT ON TXT AND FEATURES CHARGRAMS" + "\n")
of.write("kappa list: " + str(kappa_list) + "\n")
of.write("kappa mean: " + str(np.mean(kappa_list)) + "\n")
of.write("\n")
of.write("kappa sec list: " + str(kappa_sec_list) + "\n")
of.write("kappa sec mean: " + str(np.mean(kappa_sec_list)) + "\n")
of.write("\n")
of.write("classification report:" + "\n")
of.write(report_average_df.to_string())
of.write("\n")
of.write("\n")
of.write("confusion matrix:" + "\n")
for l in CM:
    of.write(str(l))
    of.write("\n")
of.write("\n")
of.write("**********************************" + "\n")
of.write("\n")
of.write("classification report sec:" + "\n")
of.write(report_average_sec_df.to_string())
of.write("\n")
of.write("\n")

print("TFIDFTVECT ON TXT AND FEATURES CHARNGRAMS is done")

of.close()
