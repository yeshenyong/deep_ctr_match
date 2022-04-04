import itertools
import random

from joblib import Parallel, delayed

from example.graph.Node2vec.alias import create_alias_table, alias_sample
from example.graph.DeepWalk.utils import partition_num


class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=0):
        """
        :params G: G data
        :params p: return parameter, controls the likelihood of immediately revisiting a node in the walk(控制在revisiting 过后才能中重新访问节点的可能性)
        :params q: In-out parameter, allows the search to differentiate between "inward" and "outward" nodes(允许搜索区分"inward" 和 "outward" 节点)
        :params use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec(node2vec 中是否使用拒绝采样策略)
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break

        return walk

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        # node2vec采用了Alias算法进行顶点采样。
        # Alias Method:时间复杂度O(1)的离散采样方法
        # 第一个采样根据权重来, 后续使用node2vec 的有偏采样
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    def node2vec_walk2(self, walk_length, start_node):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf
        """
        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if inv_p > second_upper_bound:
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))

            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))

                    while True:
                        prob = random.random() * upper_bound
                        if prob + shatter >= upper_bound:
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_nodes(alias_nodes[cur][0], alias_nodes[cur][1])]
                        if prob < lower_bound:
                            break
                        if prob < inv_p and next_node == prev:
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if prob < _prob:
                            break
                    walk.append(next_node)
            else:
                break
        return walk


    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G

        nodes = list(G.nodes())
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length)
            for num in partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
                elif self.use_rejection_sampling:
                    walks.append(self.node2vec_walk2(walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    def preprocess_trainsition_probs(self):
        """
        引导随机游动的转移概率预处理
        Preprocessing of transition probabilities for guiding the random walks
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            # node's neighbors weight list (unnormalized_probs)
            unnormalized_probs = []
            for nbr in G.neighbors(node):
                unnormalized_probs.append(G[node][nbr].get('weight', 1.0))

            norm_const = sum(unnormalized_probs)
            # normalized
            normalized_probs = []
            for u_prob in unnormalized_probs:
                normalized_probs.append(float(u_prob) / norm_const)

            alias_nodes[node] = create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}

            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
                self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return

    def get_alias_edge(self, t, v):
        """
        给出之前访问过的节点t，计算节点v与其相邻节点之间的未归一化转移概率。
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        normal_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / normal_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)
