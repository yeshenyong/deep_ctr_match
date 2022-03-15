# 推荐系统模型

笔者刚接触推荐系统未久，通过记录该库的学习更新保持

秉持着一个两周一个模型论文与实践


## Match

| 算法        | 论文    | 论文翻译与解读 |
| --------    | -----  | -----  |
| MIND            | [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030v1) | https://www.yuque.com/yeshenyong/setmh9/zbte8u |


## Rank

| 算法        | 论文    |
| :-------    | -----  |
| DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) |



## Multi-Task

| 算法        | 论文    |  公众号文章介绍  |
| --------    | -----  | ----            |
|  |  |  |


## Recall_ANN

| 算法        | 开源地址    |
| --------    | -----  |
| Annoy        | [https://github.com/spotify/annoy](https://github.com/spotify/annoy) |
|Faiss|[https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)|



### 环境配置

```shell
docker pull tensorflow/tensorflow:2.1.0
docker run --net=host  -d -t --name tensorflow  tensorflow/tensorflow:2.1.0
docker exec -it tensorflow bash
pip install deepctr[cpu]
pip install deepctr[gpu]
```







# 代码参考

> https://github.com/shenweichen/DeepCTR

> https://github.com/shenweichen/DeepMatch

> 代码样例基本基于上述开源项目进行注释更改



