"""

    Author:
        ShenYong.Ye
    Time:
        2022-03-30 13:22

"""
import math

import numpy as np

from alias import create_alias_table
from ..DeepWalk.utils import preprocess_nxgraph


class LINE:
    def __init__(self, graph, embedding_size=8, negative_ratio=5, order='second'):
        """

        :param graph
        :param embedding_size
        :param negative_ratio
        :param order: 'first', 'second', 'all'

        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be first, second, or all')
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = True

        self.rep_size = embedding_size
        self.order = order

        self._embeddings = {}
        self.negative_ratio = negative_ratio

        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        # Question
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)

        self._gen_sampling_table()
        self.reset_model()

    def _gen_sampling_table(self):
        # create sampling table for vertex
        power = 0.75
        numNodes = self.node_size
        node_degree = np.zeros(numNodes)    # out degree
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        # Question
        total_num = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[i], power)) / total_num for i in range(numNodes)]

        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        numEdges = self.graph.number_of_edges()
        # Question
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0) for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) * numEdges / total_sum for edge in self.graph.edges()]
        # Question
        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

