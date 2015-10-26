import pandas as pd
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import pytz
import numpy as np
from pprint import pprint
import random
import sys
import time

from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, classification_report, log_loss, roc_curve, auc


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs

# Returns value if key exists, and " " if not.
# Takes searched key and dictionary as input.
def getit(item, store):
    if item in store:
        return store.get(item)
    else:
        return " "


#takes file name as input, returns reformated data in three lists?
def reformat_data(file_name):
    with open(file_name, "r") as infile:
        initial_data = []
        for line in infile:
            if line == "\n":
                continue
            tmp = {}
            tmp["label"] = line.split()[0]
            for item in line.split("|")[1:]:
                tmp[item[0]] = item[2:].rstrip()
            #tmp.update({item[0]: item[2:].rstrip() for item in line.split("|")[1:]})
            if tmp.get("t"):
                tmp["t"] = tmp["t"]
            initial_data.append(tmp)


    #shuffles the intial data and pulls out the label in order.
    random.shuffle(initial_data)
    labels = []
    for item in initial_data:
        label = ""
        if item["label"] == '-1.0':
            label = 0
        if item["label"] == '1.0':
            label = 1
        labels.append(label)
        del item["label"]

    #takes the formated dataset and creates a new list of dictionaries with interations.
    # data_feature_interaction = []
    # for line in initial_data:
    #     temp_dict = line.copy()
    #     tmp = {
    #         "si": getit("s", line) + " " + getit("i", line),
    #         "pi": getit("p", line) + " " + getit("i", line),
    #         "mi": getit("m", line) + " " + getit("i", line),
    #         "ai": getit("a", line) + " " + getit("i", line),
    #         "ps": getit("p", line) + " " + getit("s", line),
    #         "ei": getit("e", line) + " " + getit("i", line),
    #         "ri": getit("r", line) + " " + getit("i", line),
    #         "pc": getit("p", line) + " " + getit("c", line),
    #         "pb": getit("p", line) + " " + getit("b", line),
    #         "bi": getit("b", line) + " " + getit("i", line),
    #         "ki": getit("k", line) + " " + getit("i", line),
    #         "pk": getit("p", line) + " " + getit("k", line),
    #         "wi": getit("w", line) + " " + getit("i", line),
    #     }
    #     temp_dict.update(tmp)
    #     data_feature_interaction.append(temp_dict)

    return (labels, initial_data)


def prediction_logistic_regression(labels, data_feature_interaction, N=10000):

    predition_type = "Logistic Regression"
    M = int(N*0.8)
    Y = np.array(labels[:N])
    feat = data_feature_interaction[:N]
    v = DictVectorizer(sparse=True)

    X = v.fit_transform(feat)

    feature_names = {}
    for name, index in v.vocabulary_.items():
        feature_names[name] = index
    #feature_names = {name: index for name, index in v.vocabulary_.items()}
    vec_X = X.toarray()
    # add this to oputput file
    #print(size of feature set:, vec_X[0])
    X_train, Y_train = vec_X[:M], Y[:M]
    X_test, Y_test = vec_X[M:], Y[M:]

    # logistic regression baseline
    lr_model = LogisticRegression(penalty='l2')
    lr = lr_model.fit(X_train, Y_train)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)
    cr = classification_report(Y_test, y_pred)
    ll = log_loss(Y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(Y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    answer = [predition_type, cr, ll, roc_auc]
    tags = [
        "predition_type: ",
        "classification_report: ",
        "log_loss: ",
        "roc_auc: "
        ]

    generate_learning_curves(lr_model, X_train, Y_train)

    return zip(tags, answer)


def prediction_random_forest(labels, data_feature_interaction, N=10000):
    predition_type = "Random Forest"
    M = int(N*0.8)
    Y = np.array(labels[:N])
    feat = data_feature_interaction[:N]
    v = DictVectorizer(sparse=True)

    X = v.fit_transform(feat)
    feature_names = {}
    for name, index in v.vocabulary_.items():
        feature_names[name] = index
    #feature_names = {name: index for name, index in v.vocabulary_.items()}
    vec_X = X.toarray()

    X_train, Y_train = vec_X[:M], Y[:M]
    X_test, Y_test = vec_X[M:], Y[M:]

    # random forest
    ensemble_model = RandomForestClassifier(n_estimators=100)
    ensemble = ensemble_model.fit(X_train, Y_train)
    en_pred = ensemble.predict(X_test)
    en_prob = ensemble.predict_proba(X_test)

    cr = classification_report(Y_test, en_pred)
    ll = log_loss(Y_test, en_prob)
    fpr, tpr, thresholds = roc_curve(Y_test, en_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    answer = [predition_type, cr, ll, roc_auc]
    tags = [
        "predition_type: ",
        "classification_report: ",
        "log_loss: ",
        "roc_auc: "
        ]

    generate_learning_curves(lr_model, X_train, Y_train)

    return zip(tags, answer)


from matplotlib.backends.backend_pdf import PdfPages
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn import cross_validation


def make_plot(x_axis, train_scores, test_scores, metric, log=False):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(x_axis, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(x_axis, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')

    if not log:
        plt.plot(x_axis, train_scores_mean, "o-", color="r",
                 label="Training {} score".format(metric))
        plt.plot(x_axis, test_scores_mean, "o-", color="g",
                 label="Validation {} scores".format(metric))
    else:
        plt.semilogx(x_axis, train_scores_mean, "o-", color="r",
                     label="Training {} score".format(metric))
        plt.semilogx(x_axis, test_scores_mean, "o-", color="g",
                     label="Validation {} scores".format(metric))
    plt.legend(loc="best")
    return plt


def plot_learning_curves(estimator, title, X, y, ylim=None, metric=None,
                         train_sizes=np.linspace(0.5, 1.0, 5)):

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    cv = cross_validation.StratifiedKFold(y, n_folds=5)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            n_jobs=1,
                                                            train_sizes=train_sizes,
                                                            scoring=metric)

    return make_plot(train_sizes, train_scores, test_scores, metric)


def generate_learning_curves(estimator, predictors, labels):
    list_metric = ["accuracy", "precision", "recall"]
    for metric in list_metric:
        title = "{} Learning Curves".format(metric.title())
        lc_plot = plot_learning_curves(estimator, title, predictors, labels, metric=metric)

        #curve_name = "learning_curves"
        pp = PdfPages("{}.pdf".format(title))
        lc_plot.savefig(pp, format='pdf')
        pp.close()


if __name__ == "__main__":

    # argunments: file_name, (logistic_regression, random_forest or both), N, output_file_name
    file_name = sys.argv[1]
    N = int(sys.argv[3])

    output_file_name = str(sys.argv[3])

    with Timer() as t:
        labels, data_feature_interaction = reformat_data(file_name)
    print "=> elasped time: %s s" % t.secs

    out_file = open("output_file_name", "w")
    if sys.argv[2] == 'logistic_regression':
        with Timer() as t:
            output = prediction_logistic_regression(labels, data_feature_interaction)
            for item in output:
                out_file.write(str(item))
                out_file.write("\n")
        print "=> elasped time: %s s" % t.secs

    if sys.argv[2] == 'random_forest':
        output = prediction_random_forest(labels, data_feature_interaction)
        for item in output:
            out_file.write(str(item))
            out_file.write("\n")

    if sys.argv[2] == 'both':
        output = prediction_random_forest(labels, data_feature_interaction)
        for item in output:
            out_file.write(str(item))
            out_file.write("\n")

        output = prediction_logistic_regression(labels, data_feature_interaction)
        for item in output:
            out_file.write(str(item))
            out_file.write("\n")

    out_file.close()
