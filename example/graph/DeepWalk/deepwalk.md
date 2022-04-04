# DeepWalk



虽然DeepWalk是KDD 2014的工作，但却是我们了解Graph Embedding无法绕过的一个方法。



我们都知道在NLP任务中，word2vec是一种常用的word embedding方法，word2vec通过语料库中的句子序列来描述词与词的共现关系，进而学习到词语的向量表示。

DeepWalk的思想类似word2vec，使用**图中节点与节点的共现关系**来学习节点的向量表示。那么关键的问题就是如何来描述节点与节点的共现关系，DeepWalk给出的方法是使用随机游走(RandomWalk)的方式在图中进行节点采样。

RandomWalk是一种**可重复访问已访问节点的深度优先遍历**算法。给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，重复此过程，直到访问序列长度满足预设条件。

获取足够数量的节点访问序列后，使用skip-gram model 进行向量学习。

![img](https://pic1.zhimg.com/80/v2-fdcb0babe4a168df85243b548fd86d30_1440w.jpg)

DeepWalk算法主要包括两个步骤，第一步为随机游走采样节点序列，第二步为使用skip-gram modelword2vec学习表达向量。

①构建同构网络，从网络中的每个节点开始分别进行Random Walk 采样，得到局部相关联的训练数据； ②对采样数据进行SkipGram训练，将离散的网络节点表示成向量化，最大化节点共现，使用Hierarchical Softmax来做超大规模分类的分类器

这里简单的用DeepWalk算法在wiki数据集上进行节点分类任务和可视化任务。 wiki数据集包含 2,405 个网页和17,981条网页之间的链接关系，以及每个网页的所属类别。

## 分类任务结果

micro-F1 : 0.6674

macro-F1 : 0.5768





















