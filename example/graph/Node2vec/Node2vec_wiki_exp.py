import numpy

import numpy as np

# from ..DeepWalk.utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from example.graph.DeepWalk.classify import Classifier, read_node_label
from node2vec import Node2Vec

import matplotlib.pyplot as plt
import networkx as nx


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f} % nodes...".format(tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings):
    X, Y = read_node_label("../data/wiki_labels.txt")
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


def print_embeddings(embeddings):
    fd = open('./embeddings.txt', 'w')
    id = sorted(embeddings.keys(), key=lambda item: int(item))
    for key in id:
        s = key + " ["
        s = s + ",".join(np.array(embeddings[key], dtype=numpy.str)) + "]\n"
        fd.write(s)
    fd.close()


if __name__ == '__main__':
    G = nx.read_edgelist('../data/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # DeepWalk each sample 10, exec 80 DeepWalk_nums
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    # window_size with list context, train for 3 epochs
    model.train(window_size=5, iter=3)

    embeddings = model.get_embeddings()
    print_embeddings(embeddings)
    evaluate_embeddings(embeddings)

    plot_embeddings(embeddings)


