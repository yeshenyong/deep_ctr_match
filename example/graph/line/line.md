# line

Larg-scale Information Network Embedding

考虑first-order proximity（local structure）和second-order proximity（global structure）实现network embeddings

之前介绍过DeepWalk，DeepWalk使用DFS随机游走在图中进行节点采样，使用word2vec在采样的序列学习图中节点的向量表示。

LINE也是一种基于邻域相似假设的方法，只不过与DeepWalk使用DFS构造邻域不同的是，LINE可以看作是一种使用BFS构造邻域的算法。此外，LINE还可以应用在带权图中(DeepWalk仅能用于无权图)。

之前还提到不同的graph embedding方法的一个主要区别是对图中顶点之间的相似度的定义不同，所以先看一下LINE对于相似度的定义。

### LINE 算法原理

1. 一种新的相似度定义

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPiaYS55rpViaWJJAck9ojFfM8YibUHJhKmiaNWD28HDTWFDsSB1XIiab7jcs0Vu4RrEGt9RXicGMHLdVlDA/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)

**first-order proximity**

1阶相似度用于描述图中成对顶点之间的局部相似度，形式化描述为若![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautVHKXRIb5fpATZ7kDibrdyfPnjTiaPsicOP6dGd2HHNZz9WPcZqrnNxmuQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之间存在直连边，则边权![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaYS55rpViaWJJAck9ojFfM8vXt4KcZBWSr2FvQcM26ia1sbmficc28WFWkSBEoSRM8RHBUrAXE5HyIA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)即为两个顶点的相似度，若不存在直连边，则1阶相似度为0。如上图，6和7之间存在直连边，且边权较大，则认为两者相似且1阶相似度较高，而5和6之间不存在直连边，则两者间1阶相似度为0。

**second-order proximity**

仅有1阶相似度就够了吗？显然不够，如上图，虽然5和6之间不存在直连边，但是他们有很多相同的邻居顶点(1,2,3,4)，这其实也可以表明5和6是相似的，而2阶相似度就是用来描述这种关系的。形式化定义为，令![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliaut3PFS60DoToeFlgraHcoXer7Dyf0OnH9ib9oC249IGjb97a3bC9J0pww/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)表示顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautia5dfATOLg2o4rz0bTuIIgHQHWcXrDjapU9ia9e9wIa2SZpJ7pp8HmyQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)与所有其他顶点间的1阶相似度，则![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautWeXfowuKD3xrQvo396JpLUQt4ogxicjYvMg0DpL6c4mlkVgkxGzcMQA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)与![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliaut8yXIp7k1TAYPLof6pYL34CQm8bomvSUbXn1OrBEVLTickjwcTBz7S4Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的2阶相似度可以通过![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautfg4EFqibQQsuFIPws69pnpphp77qujD2aDfwzv8ruGNqWKeViaeOp7gw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautuVZiaryicUvAwWWoMx4vVQNLdN4hGYoiaicfeSVypibSeHfZBHJZRV8Qmnw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的相似度表示。若![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautcxMtgdE2pI1F4Zk4LfgdGCD2sOQjHo8HOWibpzkdnF9K6a9hPalO0GQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)与![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliaut0Z7buB9Wn23YfI47nLhg02cBgwdVYFGJRRMrlVWf5jj9fN8J0nm2sg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之间不存在相同的邻居顶点，则2阶相似度为0。



2. 优化目标

1st-order

对于每一条无向边![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautBRLe4elnIp8t4OyzJUbODQrR6Qe4REsugzEfibBXJuuSwib1ltwcfRqw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，定义顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautcicVqo1afAJ7rtibhbcb0tCRhyicLD2VykicFpuSoUP5dexcCTEW2eLqow/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautnuTLEFEWCaKlYJmyCYicj4DlV1fSMq7mlzkcQq5sE7JkBE2dUtA8rLw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之间的联合概率为：

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautn4KMY0suYPDFtVEmdapSTlnnQOhg57hnhQ2icZL06j0IGfVyNibIcETg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautZcpEVSrxT7UzaR5cN2ia5FmCHO1vY15okvIoaJec59tEmI8FJXPdyYg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)为顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautxia0wV9Ccd9nludopDhmeticMVS856Rv8reV3KgKorcjiag0D9Pz8emUw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的低维向量表示。（可以看作一个内积模型，计算两个item之间的匹配程度）

同时定义经验分布：

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliaut7MK0K5USGSD1nZdffZN0YgtUY0KicABoHHhJK9dpfAd2OHY6mWcgib0A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1), 

优化目标为最小化：

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautwoTRjJJ4KuEYM9iaxicOMh72Io2mbyIBA8ibGanuh8BicbQD0NXiatemgcg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliaut3vAibyIxcfeSiaibZkria0PTjdCNenV8jVn4ib0ny52l3K0JC0cZc71X84Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是两个分布的距离，常用的衡量两个概率分布差异的指标为KL散度，使用KL散度并忽略常数项后有：

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiaXyZEz3VbuoTDGNclFliautiaLHc4mlK2S1Ma293o4icruZMEfIDQ2WNPee58pUsGHdV1tSLAncp1fg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**1st order 相似度只能用于无向图当中。**

**2nd order**





https://blog.csdn.net/weixin_37688445/article/details/104106781



























