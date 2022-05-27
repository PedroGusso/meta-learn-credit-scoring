import pandas as pd
import numpy as np
from psi import calculate_psi
from scipy.stats import ks_2samp

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from hypothesis_testing_meta import omnibus_posthoc

# Table with statistics
table = []

# Names and methods of classifiers
names = ['Min', 'Max', 'Average', 'Weighted Average']

# Matrices for doing the friedman and nemenyi tests
friedman_nemenyi_ks = pd.DataFrame(columns=names)
friedman_nemenyi_auc = pd.DataFrame(columns=names)
friedman_nemenyi_f1 = pd.DataFrame(columns=names)
friedman_nemenyi_psi = pd.DataFrame(columns=names)


def home_credit_test():
    # Experiment with the dataset home-credit-default-risk
    data = pd.read_csv('meta-datasets/home-credit-default-risk_test.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def home_credit_train():
    # Experiment with the dataset home-credit-default-risk
    data = pd.read_csv('meta-datasets/home-credit-default-risk_train.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    return X, y


def give_credit_test():
    # Experiment with the dataset Give Me Some Credit
    data = pd.read_csv('meta-datasets/Give Me Some Credit_test.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def give_credit_train():
    # Experiment with the dataset Give Me Some Credit
    data = pd.read_csv('meta-datasets/Give Me Some Credit_train.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    return X, y


def taiwanese_test():
    # Experiment with the dataset taiwanese
    data = pd.read_csv('meta-datasets/Taiwanese_test.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def taiwanese_train():
    # Experiment with the dataset taiwanese
    data = pd.read_csv('meta-datasets/Taiwanese_train.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    return X, y


def australian_test():
    # Experiment with the dataset australian
    data = pd.read_csv('meta-datasets/Australian_test.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def australian_train():
    # Experiment with the dataset australian
    data = pd.read_csv('meta-datasets/Australian_train.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    return X, y


def german_test():
    # Experiment with the dataset german
    data = pd.read_csv('meta-datasets/German_test.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def german_train():
    # Experiment with the dataset german
    data = pd.read_csv('meta-datasets/German_train.csv')
    X, y = data.iloc[:, 2:-1], data.iloc[:, -1:]
    return X, y


def stats(X_test, y_test, dataset_label, X_train, y_train, dataset_name):
    # List of KS, AUC, F1 and PSI results to insert in the dataframe that will be used for
    # calculate friedman and nemenyi tests
    ks_list = []
    auc_list = []
    f1_list = []
    psi_list = []

    # List with detection rate plots
    dr_list_plot = []

    X_train_KS = pd.concat([X_train, y_train], axis=1).to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    X_train = X_train.to_numpy()

    for name in names:
        if name == 'Min':
            y_proba_method = np.amin(X_test, axis=1)
            y_pred_method = np.around(y_proba_method, decimals=0)
            y_proba_train = np.amin(X_train, axis=1)
        elif name == 'Max':
            y_proba_method = np.amax(X_test, axis=1)
            y_pred_method = np.around(y_proba_method, decimals=0)
            y_proba_train = np.amax(X_train, axis=1)
        elif name == 'Average':
            y_proba_method = np.average(X_test, axis=1)
            y_pred_method = np.around(y_proba_method, decimals=0)
            y_proba_train = np.average(X_train, axis=1)
        else:
            weight_ks = []
            for counter in range(4):
                ks = ks_2samp(X_train_KS[np.where(X_train_KS[:, 4] == dataset_label)][:, counter],
                              X_train_KS[np.where(X_train_KS[:, 4] != dataset_label)][:, counter])
                weight_ks.append(ks[0])
            y_proba_method = np.average(X_test, axis=1, weights=weight_ks)
            y_pred_method = np.around(y_proba_method, decimals=0)
            y_proba_train = np.average(X_train, axis=1, weights=weight_ks)

        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_proba_method, pos_label=dataset_label)
        plt.plot(fpr_roc, tpr_roc)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_method).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        auc = roc_auc_score(y_test, y_proba_method)
        f1 = f1_score(y_test, y_pred_method, average=None)

        # ks with the positive and negative class probabilities
        ks = ks_2samp(y_proba_method[y_test[:, 0] == dataset_label],
                      y_proba_method[y_test[:, 0] != dataset_label])
        ks_result = ks[0]
        ks_pvalue = ks[1]

        y_pred_method_flipped = np.flip(y_pred_method)
        y_proba_method_flipped = np.flip(y_proba_method)

        # Uninformed. Replace y_test with training probabilities
        # Make a histogram of the two variables passed as a parameter
        psi = calculate_psi(y_proba_train, y_proba_method_flipped, axis=1)

        # See if it's not inverted, get the KS complement, 1 - the value of KS
        # Add the current ks, auc, f1 score and psi to the list to do omnibus and post hoc testing
        ks_list.append(ks_result)
        auc_list.append(auc)
        f1_list.append(f1[1])
        psi_list.append(psi)

        # Table of statistical results
        # I just get the f1 score from the positive class
        table.append([dataset_test, name, fpr, fnr, auc, f1[1], ks_result, ks_pvalue, psi])

        # Do the detection rate (sensitivity) as follows: Sort a matrix with 3 columns (proba, pred)
        # Sort by proba and divide this 10% matrix by 10% making a detection rate point for each division
        # Plot the graph with all points

        y_proba_method_percent = [x * 100 for x in y_proba_method]
        dr_matrix = np.array((y_proba_method_percent, y_pred_method_flipped))
        dr_matrix = np.transpose(dr_matrix)
        dr_matrix = np.flip(np.sort(dr_matrix, axis=0), axis=0)
        dr_groups = np.array_split(dr_matrix, 10, axis=0)
        dr_list = []
        cumulative_dr = 0
        for i in range(len(dr_groups)):
            # all positives in the subgroup divided by the total number of positives in the entire base (cumulative)
            # Since DR is over all positives, I just divided by all positives
            # Try block except if it has division by zero
            try:
                cumulative_dr += len([x for x in dr_groups[i][:, 1] if x == dataset_label]) / len(
                    [i for i in dr_matrix[:, 1] if i == dataset_label])
            except:
                cumulative_dr += 0
            dr_list.append(cumulative_dr)
        dr_list_plot.append(dr_list)

    # Plot the ROC graph
    plt.title('Receiver Operating Characteristic: ' + dataset_name)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(names)
    plt.savefig('results_meta/ROC-' + dataset_test)
    plt.clf()

    # Plot the DR graph
    plt.title('Cumulative Detection Rate X Test Subgroups: ' + dataset_name)
    plt.xlabel('Test Subgroups')
    plt.ylabel('Cumulative Detection Rate')
    for p, name in zip(dr_list_plot, names):
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p, marker='o')
    plt.grid(True)
    plt.xticks(np.arange(1, 11, step=1))
    plt.legend(names)
    plt.savefig('results_meta/DR-' + dataset_test)
    plt.clf()

    # Put the KS, AUC, F1 and PSI lists in the dataframes
    friedman_nemenyi_ks.loc[len(friedman_nemenyi_ks)] = ks_list
    friedman_nemenyi_auc.loc[len(friedman_nemenyi_auc)] = auc_list
    friedman_nemenyi_f1.loc[len(friedman_nemenyi_f1)] = f1_list
    friedman_nemenyi_psi.loc[len(friedman_nemenyi_psi)] = psi_list


if __name__ == '__main__':
    # Header of the dataframe that will be saved in a file with the statistics
    header = ['Dataset', 'Classifiers', 'FPR', 'FNR', 'AUC', 'F1 Score', 'KS Result', 'KS p-value', 'PSI']
    datasets_dict_test = {
        "Australian": australian_test(),
        "German": german_test(),
        "Taiwanese": taiwanese_test(),
        "Give Me Some Credit": give_credit_test(),
        "home-credit-default-risk": home_credit_test(),
    }
    datasets_dict_train = {
        "Australian": australian_train(),
        "German": german_train(),
        "Taiwanese": taiwanese_train(),
        "Give Me Some Credit": give_credit_train(),
        "home-credit-default-risk": home_credit_train(),
    }
    # tqdm in looping to show algorithm progress bar
    for dataset_test, dataset_train in tqdm(zip(datasets_dict_test, datasets_dict_train)):
        # get the test's target label
        test_X, test_y, target_label = datasets_dict_test[dataset_test]
        train_X, train_y = datasets_dict_train[dataset_train]
        # get the dataset name by test
        stats(test_X, test_y, target_label, train_X, train_y, dataset_test)
    stats_table = pd.DataFrame(data=table, columns=header)
    stats_table.to_csv("results_blend/metrics.csv")
    omnibus_posthoc(friedman_nemenyi_ks, "ks", "results_meta")
    omnibus_posthoc(friedman_nemenyi_auc, "auc", "results_meta")
    omnibus_posthoc(friedman_nemenyi_f1, "f1", "results_meta")
    omnibus_posthoc(friedman_nemenyi_psi, "psi", "results_meta")
