"""
    Author:
        1050575224@qq.com

    Reference:
        [1] DeepWalk
"""

from RandomWalker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks

        self.walker = RandomWalker(graph, p=1, q=1)
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1    # skip gram
        kwargs["hs"] = 1    # deepwalk use Hierarchical softmax(超分类)
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vector done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


