# Struc2Vec

参考自：

1. https://zhuanlan.zhihu.com/p/56733145

​	其中DeepWalk,Node2Vec通过随机游走在图中采样顶点序列来构造顶点的近邻集合。LINE显式的构造邻接点对和顶点的距离为1的近邻集合。SDNE使用邻接矩阵描述顶点的近邻结构。

​	事实上，在一些场景中，两个不是近邻的顶点也可能拥有很高的相似性，对于这类相似性，上述方法是无法捕捉到的。Struc2Vec就是针对这类场景提出的。Struc2Vec的论文发表在2017年的KDD会议中。



## Struc2Vec算法原理

### 相似度定义

Struc2Vec是从空间结构相似性的角度定义顶点相似度的。

用下面的图简单解释下，如果在基于近邻相似的模型中，顶点u和顶点v是不相似的，第一他们不直接相连，第二他们不共享任何邻居顶点。

而在struc2vec的假设中，顶点u和顶点v是具有空间结构相似的。他们的度数分别为5和4，分别连接3个和2个三角形结构，通过2个顶点(d,e;x,w)和网络的其他部分相连。

![img](https://pic1.zhimg.com/80/v2-1b5d443aa09dbb3fa29f31634a2357c4_1440w.jpg)

直观来看，具有相同度数的顶点是结构相似的，若各自邻接顶点仍然具有相同度数，那么他们的相似度就更高。



### 顶点对距离定义

令 ![[公式]](https://www.zhihu.com/equation?tex=R_k%28u%29) 表示到顶点u距离为k的顶点集合，则 ![[公式]](https://www.zhihu.com/equation?tex=R_1%28u%29) 表示是u的直接相连近邻集合。

令 ![[公式]](https://www.zhihu.com/equation?tex=s%28S%29) 表示顶点集合S的**有序度序列**。

通过比较两个顶点之间距离为k的环路上的有序度序列可以推出一种层次化衡量结构相似度的方法。

令 ![[公式]](https://www.zhihu.com/equation?tex=f_k%28u%2Cv%29) 表示顶点u和v之间距离为k（这里的距离k实际上是指距离小于等于k的节点集合）的环路上的结构距离(注意是距离，不是相似度)。

![[公式]](https://www.zhihu.com/equation?tex=f_k%28u%2Cv%29%3Df_%7Bk-1%7D%28u%2Cv%29%2Bg%28s%28R_k%28u%29%29%2Cs%28R_k%28v%29%29%29%2Ck%5Cge+0+%5Ctext%7B+and+%7D+%7CR_k%28u%29%7C%2C%7CR_k%28v%29%7C%3E0)

其中 ![[公式]](https://www.zhihu.com/equation?tex=g%28D_1%2CD_2%29%5Cge+0) 是衡量有序度序列 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_2) 的距离的函数，并且 ![[公式]](https://www.zhihu.com/equation?tex=f_%7B-1%7D%3D0) .

下面就是如何定义有序度序列之间的比较函数了，由于 ![[公式]](https://www.zhihu.com/equation?tex=s%28R_k%28u%29%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=s%28R_k%28v%29%29) 的长度不同，并且可能含有重复元素。所以文章采用了**Dynamic Time Warping(DTW)**来衡量两个有序度序列。

一句话，DTW可以用来衡量两个不同长度且含有重复元素的的序列的距离（距离的定义可以自己设置）。

基于DTW，定义元素之间的距离函数 ![[公式]](https://www.zhihu.com/equation?tex=d%28a%2Cb%29%3D%5Cfrac%7Bmax%28a%2Cb%29%7D%7Bmin%28a%2Cb%29%7D-1)

至于为什么使用这样的距离函数，这个距离函数实际上惩罚了当两个顶点的度数都比较小的时候两者的差异。举例来说， ![[公式]](https://www.zhihu.com/equation?tex=a%3D1%2Cb%3D2) 情况下的距离为1， ![[公式]](https://www.zhihu.com/equation?tex=a%3D101%2Cb%3D102) 情况下的距离差异为0.0099。这个特性正是我们想要的。





### 构建层次带权图

使用有偏随机游走在构造出的图 ![[公式]](https://www.zhihu.com/equation?tex=M) 中进行顶点序列采样。 每次采样时，首先决定是在当前层游走，还是切换到上下层的层游走。

若决定在当前层游走，设当前处于第k层，则从顶点u到顶点v的概率为：

![[公式]](https://www.zhihu.com/equation?tex=p_k%28u%2Cv%29%3D%5Cfrac%7Be%5E%7B-f_k%28u%2Cv%29%7D%7D%7BZ_k%28u%29%7D) 其中 ![[公式]](https://www.zhihu.com/equation?tex=Z_k%28u%29%3D%5Csum_%7Bv%5Cin+V%2Cv%5Cne+u%7De%5E%7B-f_k%28u%2Cv%29%7D) 是第k层中关于顶点u的归一化因子。

通过在图M中进行随机游走，每次采样的顶点更倾向于选择与当前顶点结构相似的顶点。因此，**采样生成的上下文顶点很可能是结构相似的顶点，这与顶点在图中的位置无关**。

若决定切换不同的层，则以如下的概率选择 ![[公式]](https://www.zhihu.com/equation?tex=k%2B1) 层或 ![[公式]](https://www.zhihu.com/equation?tex=k-1) 层，

![[公式]](https://www.zhihu.com/equation?tex=p_k%28u_k%2Cu_%7Bk%2B1%7D%29%3D%5Cfrac%7Bw%28u_k%2Cu_%7Bk%2B1%7D%29%7D%7Bw%28u_k%2Cu_%7Bk%2B1%7D%29%2Bw%28u_k%2Cu_%7Bk-1%7D%29%7D)

![[公式]](https://www.zhihu.com/equation?tex=p_k%28u_k%2Cu_%7Bk-1%7D%29%3D1-p_k%28u_k%2Cu_k%2B1%29)



### 三个时空复杂度优化技巧

#### OPT1 有序度序列长度优化

前面提到过对于每个顶点在每一层都有一个有序度序列，而每一个度序列的空间复杂度为O(n)。

文章提出一种压缩表示方法，对于序列中出现的每一个度，计算该度在序列里出现的次数。压缩后的有序度序列存储的是**(度数，出现次数)**这样的二元组。

同时修改距离函数为： ![[公式]](https://www.zhihu.com/equation?tex=dist%28a%2Cb%29%3D%28%5Cfrac%7Bmax%28a_0%2Cb_0%29%7D%7Bmin%28a_0%2Cb_0%29%7D-1%29max%28a_1%2Cb_1%29) ![[公式]](https://www.zhihu.com/equation?tex=a_0%2Cb_0) 为度数， ![[公式]](https://www.zhihu.com/equation?tex=a_1%2Cb_1) 为度的出现次数。



#### OPT2 相似度计算优化

在原始的方法中，我们需要计算每一层k中，任意两个顶点之间的相似度。事实上，这是不必要的。因为两个度数相差很大的顶点，即使在 ![[公式]](https://www.zhihu.com/equation?tex=k%3D0) 的时候他们的距离也已经非常大了，那么在随机游走时这样的边就几乎不可能被采样到，所以我们也没必要计算这两个顶点之间的距离。

文章给出的方法是在计算顶点u和其他顶点之间的距离时，只计算那些与顶点u的度数接近的顶点的距离。具体来说，在顶点u对应的有序度序列中进行二分查找，查找的过程就是不断逼近顶点u的度数的过程，只计算查找路径上的顶点与u的距离。 这样每一次需要计算的边的数量从 ![[公式]](https://www.zhihu.com/equation?tex=n%5E2) 数量级缩减到 ![[公式]](https://www.zhihu.com/equation?tex=n%5Clog%7Bn%7D) 



#### OPT3 限制层次带权图层数

层次带权图M中的层数是由图的直径 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%2A) 决定的。但是对很多图来说，图的直径会远远大于顶点之间的平均距离。

当k接近 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%2A) 时，环上的度序列 ![[公式]](https://www.zhihu.com/equation?tex=s%28R_k%28u%29%29) 长度也会变得很短， ![[公式]](https://www.zhihu.com/equation?tex=f_k%28u%2Cv%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bk-1%7D%28u%2Cv%29) 会变得接近。

因此将图中的层数限制为 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7B%27%7D%3Ck%5E%2A) ，使用最重要的一些层来评估结构相似度。

这样的限制显著降低构造M时的计算和存储开销。



### Struc2Vec核心代码

Struc2Vec的实现相比于前面的几个算法稍微复杂一些，这里我主要说下大体思路，对一些细节有疑问的同学可以邮件或者私信我~

根据前面的算法原理介绍，首先确定一下我们要做哪些事情

1. 获取每一层的顶点对距离 
2. 根据顶点对距离构建带权层次图 
3. **在带权层次图中随机游走采样顶点序列**



#### 随机游走采样

采样的主体框架和前面的DeepWalk,Node2Vec差不多，这里就说下不同的地方。 由于Struc2Vec是在一个多层图中进行采样，游走可能发生在同一层中，也可能发生跨层，所以要添加一些跨层处理的逻辑。

![[公式]](https://www.zhihu.com/equation?tex=p_k%28u_k%2Cu_%7Bk%2B1%7D%29%3D%5Cfrac%7Bw%28u_k%2Cu_%7Bk%2B1%7D%29%7D%7Bw%28u_k%2Cu_%7Bk%2B1%7D%29%2Bw%28u_k%2Cu_%7Bk-1%7D%29%7D)

![[公式]](https://www.zhihu.com/equation?tex=p_k%28u_k%2Cu_%7Bk-1%7D%29%3D1-p_k%28u_k%2Cu_k%2B1%29)







### Struc2Vec应用

Struc2Vec应用于无权无向图(带权图的权重不会用到，有向图会当成无向图处理)，主要关注的是图中顶点的空间结构相似性，这里我们采用论文中使用的一个数据集。该数据集是一个机场流量的数据集，顶点表示机场，边表示两个机场之间存在航班。机场会被打上活跃等级的标签。































