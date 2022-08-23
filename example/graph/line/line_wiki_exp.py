import numpy
import numpy as np

from LINE import *
from ..DeepWalk.utils import *
from ..DeepWalk.classify import *
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
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
    for key, value in embeddings.items():
        s = key + " ["
        s = s + ",".join(np.array(value, dtype=numpy.str)) + "]\n"
        fd.write(s)
    fd.close()


if __name__ == '__main__':
    G = nx.read_edgelist('../data/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # LINE
    model = LINE(G, embedding_size=128, order='second')

    model.train(batch_size=1024, epochs=50, verbose=2)

    embeddings = model.get_embeddings()
    print_embeddings(embeddings)
    evaluate_embeddings(embeddings)

    plot_embeddings(embeddings)








