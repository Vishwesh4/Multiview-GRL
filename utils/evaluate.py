# --------------------------------------------------------------------------------------------------------------------------
# Description: This script is about evaluating the learnt representation using DGI and replicating results of the paper
# The code on evaluation of model has been taken from https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/evaluate_embedding.py
# --------------------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Linear Regression
def train_logistic_classify(x, y, device):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    acc = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = LogisticRegression(random_state=42)
        classifier.fit(x_train, y_train)
        acc.append(accuracy_score(y_test,classifier.predict(x_test)))
    ret = np.mean(acc)
    return ret

def val_logistic_classify(x_train, y_train, x_test, device):
    classifier = LogisticRegression(random_state=42)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    return torch.tensor(pred).to(device)

## TSNE Map
def generate_tsnemaps(features, labels, legend):
    #T-SNE visualization
    n_classes = len(np.unique(labels))
    tsne = TSNE()
    Y = tsne.fit_transform(features)
    fig = plt.figure(figsize=(15, 15))
    colors = np.array(sns.color_palette(None,n_classes))
    target_names = list(np.arange(0,n_classes))
    target_ids = range(len(target_names))
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(Y[np.where(labels==i)[0], 0], Y[np.where(labels==i)[0], 1], c=c, label=legend[label])
    plt.legend()
    return fig
