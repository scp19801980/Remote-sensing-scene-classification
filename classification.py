import pandas as pd 
import seaborn as sn
import numpy as np 
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
#csv_name:using predict.py and the csv file will be generated.
#labels:A list. The list includes all the category names of the dataset being tested.
#csv_name2:using train.py and the csv file will be gernerated.
def acc_score(csv_name):
    r_c = pd.read_csv('./result_show/' + csv_name)
    true_labels = r_c['true_labels']
    pred_labels = r_c['pred_labels']
    acc = accuracy_score(true_labels, pred_labels)
    return acc

def report(csv_name, labels):
    r_c = pd.read_csv('./result_show/' + csv_name)
    true_labels = r_c['true_labels']
    pred_labels = r_c['pred_labels']
    r = classification_report(true_labels, pred_labels, digits=4, target_names=labels)
    return r

def matrix(csv_name, labels):
    r_c = pd.read_csv('./result_show/' + csv_name)
    true_labels = r_c['true_labels']
    pred_labels = r_c['pred_labels']
    mat = confusion_matrix(true_labels, pred_labels)
    mat_2 = np.ndarray((len(labels), len(labels)))
    names = []
    for n in range(1, len(labels)+1):
        name = str(n) + '#'
        names.append(name)

    for i in range(len(labels)):
        for k in range(len(labels)):
            mat_2[i][k] = mat[i][k] / np.sum(mat[i])

    mat_2 = np.round(mat_2, decimals=2)
    sn.heatmap(mat_2, annot=True, fmt='.2f', cmap='gray_r', xticklabels=names, yticklabels=labels,
            mask=mat_2<0.001, annot_kws={'size':8})
    plt.yticks(rotation=360)
    plt.show()

def plt_acc(csv_name2):
    r_c = pd.read_csv(csv_name2)
    acc = r_c['acc']
    val_acc = r_c['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'blue', label='train_acc', marker='', linestyle='-')
    plt.plot(epochs, val_acc, 'red', label='test_acc', marker='.', linestyle='-')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

def plt_loss(csv_name2):
    r_c = pd.read_csv(csv_name2)
    loss = r_c['loss']
    val_loss = r_c['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'blue', label='train_loss', marker='', linestyle='-')
    plt.plot(epochs, val_loss, 'red', label='test_loss', marker='.', linestyle='-')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.grid()
    plt.show()


