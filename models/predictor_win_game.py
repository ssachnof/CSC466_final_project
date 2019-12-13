import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import sys
warnings.filterwarnings('ignore')


def tn(confusion_matrix): return confusion_matrix[1][1]
def fp(confusion_matrix): return confusion_matrix[1][0]
def fn(confusion_matrix): return confusion_matrix[0][1]
def tp(confusion_matrix): return confusion_matrix[0][0]

def driver_decision_tree(data, targets, max_cols):
    overall_best_model, overall_best_metrics, overall_best_cols = None, None, None
    overall_best_threshold = None
    overall_best_cm = None
    for n_cols in range(1, max_cols + 1):
        possible_cols = itertools.combinations(data.columns, n_cols)
        possible_cols = list(possible_cols)
        # l = len(list(possible_cols))
        # print('possible cols length: ', len(possible_cols))
        k = 0
        for selected_cols in possible_cols:
            # print(k)
            k += 1
            selected_cols = list(selected_cols)
            for t in range(1, 10, 2):#cut threshold off at 50% entropy

                threshold = t / 10
                #threshold = t
                model = DecisionTreeClassifier(min_impurity_split= threshold, criterion= 'entropy')
                y_pred = cross_val_predict(model, data[selected_cols], targets, cv=10)
                conf_mat = confusion_matrix(targets, y_pred)
                num_tp, num_tn, num_fp, num_fn = tp(conf_mat), tn(conf_mat), fp(conf_mat), fn(conf_mat)
                #                 score = cross_validate(model, data[selected_cols], targets, scoring = ['accuracy', 'f1'], cv=10)

                acc = (num_tp + num_tn) / (num_tp + num_fn + num_tn + num_fp)
                prec = num_tp / (num_tp + num_fp)
                recall = num_tp / (num_tp + num_fn)
                f_measure = 2 * ((prec * recall) / (prec + recall))
                metrics = {'f1': f_measure, 'acc' : acc, 'prec':prec}
                if overall_best_model is None:
                    overall_best_cols = selected_cols
                    overall_best_metrics = metrics
                    overall_best_model = model
                    overall_best_threshold = threshold
                    overall_best_cm = conf_mat
                elif metrics['acc'] > overall_best_metrics['acc']:
                    overall_best_cols = selected_cols
                    overall_best_metrics = metrics
                    overall_best_model = model
                    overall_best_threshold = threshold
                    overall_best_cm = conf_mat
        print("n_cols = ", n_cols)
        print('overall_cm: ')
        print(overall_best_cm)
        print('overall: ', overall_best_metrics, overall_best_cols, overall_best_threshold)
def driver_random_forest(data, targets, max_cols):
    overall_best_model, overall_best_metrics, overall_best_cols = None, None, None
    overall_best_hyperparams = None
    overall_best_cm = None
    max_samples = len(data)
    for n_cols in range(4, max_cols + 1):
        possible_cols = itertools.combinations(data.columns, n_cols)
        possible_cols = list(possible_cols)
        # l = len(list(possible_cols))
        # print('possible cols length: ', len(possible_cols))
        k = 0
        for selected_cols in possible_cols:
            k += 1
            selected_cols = list(selected_cols)
            for n_trees in range(100, 300, 100):#cut threshold off at 50% entropy
                # print(n_trees)
                for n_features in range(3, 5):
                    # print(n_features)
                    #threshold = t
                    model = RandomForestClassifier(criterion='entropy', n_estimators=n_trees, bootstrap=True)
                    y_pred = cross_val_predict(model, data[selected_cols], targets, cv=10)
                    conf_mat = confusion_matrix(targets, y_pred)
                    num_tp, num_tn, num_fp, num_fn = tp(conf_mat), tn(conf_mat), fp(conf_mat), fn(conf_mat)
                    #                 score = cross_validate(model, data[selected_cols], targets, scoring = ['accuracy', 'f1'], cv=10)

                    acc = (num_tp + num_tn) / (num_tp + num_fn + num_tn + num_fp)
                    prec = num_tp / (num_tp + num_fp)
                    recall = num_tp / (num_tp + num_fn)
                    f_measure = 2 * ((prec * recall) / (prec + recall))
                    metrics = {'f1': f_measure, 'acc' : acc, 'prec':prec}

                    if overall_best_model is None:
                        overall_best_metrics = metrics
                        overall_best_model = model
                        overall_best_hyperparams = [n_trees, n_features]
                        overall_best_cm = conf_mat
                    elif metrics['acc'] > overall_best_metrics['acc']:
                        overall_best_cols = selected_cols
                        overall_best_metrics = metrics
                        overall_best_model = model
                        overall_best_hyperparams = [n_trees, n_features]
                        overall_best_cm = conf_mat
        print("n_cols = ", n_cols)
        print('overall_cm: ')
        print(overall_best_cm)
        print('overall: ', overall_best_metrics, overall_best_cols, overall_best_hyperparams)
def rf(data, num_trees, targets, selected_cols):
    model = RandomForestClassifier(criterion='entropy', n_estimators=num_trees, bootstrap=True)
    y_pred = cross_val_predict(model, data[selected_cols], targets, cv=10)
    conf_mat = confusion_matrix(targets, y_pred)
    num_tp, num_tn, num_fp, num_fn = tp(conf_mat), tn(conf_mat), fp(conf_mat), fn(conf_mat)
    acc = (num_tp + num_tn) / (num_tp + num_fn + num_tn + num_fp)
    prec = num_tp / (num_tp + num_fp)
    recall = num_tp / (num_tp + num_fn)
    f_measure = 2 * ((prec * recall) / (prec + recall))
    metrics = {'f1': f_measure, 'acc' : acc, 'prec':prec}
    print(metrics)
    print(conf_mat)
    return model

def dt(data, threshold, targets, selected_cols):
    model = DecisionTreeClassifier(min_impurity_split= threshold, criterion= 'entropy')
    y_pred = cross_val_predict(model, data[selected_cols], targets, cv=10)
    conf_mat = confusion_matrix(targets, y_pred)
    num_tp, num_tn, num_fp, num_fn = tp(conf_mat), tn(conf_mat), fp(conf_mat), fn(conf_mat)
    acc = (num_tp + num_tn) / (num_tp + num_fn + num_tn + num_fp)
    prec = num_tp / (num_tp + num_fp)
    recall = num_tp / (num_tp + num_fn)
    f_measure = 2 * ((prec * recall) / (prec + recall))
    metrics = {'f1': f_measure, 'acc' : acc, 'prec':prec}
    print(metrics)
    print(conf_mat)
    return model



def get_data(filename):
    data = pd.read_csv(filename)
    data = data.drop(columns=[data.columns[0], 'game_id', 'team', 'fumble', 'interception'])
    data['is_winner'] = data['is_winner'].replace([False, True], [0, 1])#vectorize the is_winner category
    targets = data['is_winner']
    data = data.drop(columns = ['is_winner'])
    cols = data.columns
    return data, targets


def main():
    filename = sys.argv[1]
    method = sys.argv[2]
    data, targets = get_data(filename)
    selected_cols = sys.argv[4:]
    if '-model_selection' in sys.argv:
        max_cols = int(sys.argv[2])
        method = sys.argv[3]
        if method == '-rf':
            driver_random_forest(data, targets, max_cols)
        else:
            driver_decision_tree(data, targets, max_cols)
    for i in range(len(selected_cols)):
        selected_cols[i] = selected_cols[i].strip()
    if method == '-rf':
        num_trees = int(sys.argv[3])
        rf(data, num_trees, targets, selected_cols)
    else:
        threshold = float(sys.argv[3])
        dt(data, threshold, targets, selected_cols)

if __name__ == "__main__":
    main()

