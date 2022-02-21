# 点击预估模型


## 1. Recall

| 算法        | 论文    |  公众号或知乎文章介绍  |
| --------    | -----  | ----            |
| MIND            | [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030v1) | [推荐系统召回模型之MIND用户多兴趣网络实践](https://mp.weixin.qq.com/s/Ys4EZw97ulrcBWFdN1OMyQ) |


## 2. Rank

| 算法        | 论文    |  公众号文章介绍  |
| --------    | -----  | ----            |
| FFM        | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) | [FFM算法原理及Bi-FFM算法实现](https://mp.weixin.qq.com/s/T46HbKC-_9yYzVTgl8Fh8w) |
| Wide & Deep      | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) |  |
| NFM            | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf) | [NFM模型理论与实践](https://mp.weixin.qq.com/s/1sWYlzIydiLAPMBnr-a5sQ) |
| AFM            | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf) | [注意力机制在深度推荐算法中的应用之AFM模型](https://mp.weixin.qq.com/s/sj5bxwtgiw-SaIItsjbeew) |
| DeepFM            | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | [DeepFM实践](https://zhuanlan.zhihu.com/p/137894818) |
| BST            | [Behavior sequence transformer for e-commerce recommendation in Alibaba](https://arxiv.org/pdf/1905.06874.pdf) | [Transformer 在美团搜索排序中的实践](https://zhuanlan.zhihu.com/p/161311198) |



## 3. Multi-Task

| 算法        | 论文    |  公众号文章介绍  |
| --------    | -----  | ----            |
| ESMM        | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) | [ESMM多任务学习算法在推荐系统中的应用](https://mp.weixin.qq.com/s/x521rMWLf6CLk0e2uXEJng) |
| MMoE      | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) | [多任务学习之MMOE模型](https://mp.weixin.qq.com/s/cBy0Y5xDtkc6PxhF1HNomg) |


## 4. Recall_ANN

| 算法        | 开源地址    |  公众号文章介绍  |
| --------    | -----  | ----            |
| Annoy        | [https://github.com/spotify/annoy](https://github.com/spotify/annoy) | [Annoy最近邻检索技术之 “图片检索”](https://zhuanlan.zhihu.com/p/148819536) |
|Faiss|[https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)||



# 代码参考

> https://github.com/shenweichen/DeepCTR

> https://github.com/shenweichen/DeepMatch


# 待学习及分享

## Recall

[Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://arxiv.org/abs/1905.01969v3)

[Controllable Multi-Interest Framework for Recommendation](https://static.aminer.cn/storage/pdf/arxiv/20/2005/2005.09347.pdf)，
代码：[https://github.com/THUDM/ComiRec](https://github.com/THUDM/ComiRec)

## Pre-Rank

[COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/abs/2007.16122)


## Rank

DIN：[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

DIEN：[Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)，
代码： [https://github.com/mouna99/dien](https://github.com/mouna99/dien)

MIMN：[Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09248.pdf)

Search-based Interest Model：[Search-based User Interest Modeling with Lifelong Sequential
Behavior Data for Click-Through Rate Prediction](https://arxiv.org/pdf/2006.05639.pdf)


## Multi-Task
YouTube，2019: Recommending What Video to Watch Next-A Multitask Ranking System
