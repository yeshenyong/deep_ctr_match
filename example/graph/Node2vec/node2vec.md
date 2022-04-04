# node2vec

设 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSBrVDTy6Vkicp4BiaVOczcicPicbcIyy7PVNHnIawKYSfJGaeRvfXbvfcrQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是将顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSXVMoMqib3hV1QAs6m1NgZ4zHoB5jYg2RpjJMw3sXrrkYRssqiaaicBspg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)映射为embedding向量的映射函数,对于图中每个顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSWsVFu783pXVLPcz6a7qbMCjAEqhOb8AdNA1aymJB53f0CictqVX0mDQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，定义 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSZr34jMD9qM5CgE1icD4V47N6Y7mqMWBCOjXQZvZSULxicM0J3U8FUsNg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)为通过采样策略![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSUD6iaiaeicm4Kab27vRskseOV1ZSJq3UTHAlibdcq8ZBVWF4TL1r5ic9gUg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)采样出的顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSr0iaOrrmaMx09Sv2IFBM2CKr4roJ2yUibfNbFWeDibxrEJYdgAia5E4BEA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的近邻顶点集合。





### **顶点序列采样策略**

node2vec依然采用随机游走的方式获取顶点的近邻序列，不同的是node2vec采用的是一种有偏的随机游走。

给定当前顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSNPs5yu98q9hMBSDZQt2r3Xv4HEYP2dxdRetHduXSQpP8VG4bpfQYvg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，访问下一个顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSIlx4wfZlCvkia8X4VnRhX92VuibrTHibTR5phX2q8zFHJbQiaeKGOCib1IQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的概率为：



![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSVOnRlw3AnRdbbpMmfWdanNHkdM1jtYXJ1oVWPIa2GojYOCkias4hLtQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSsKhN7icKmJY0IcmZRsI5pXIpqLoR47DcBxE0ernNQdRMSZqOkyVs5kw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSClzJUm4iagmwFAdhC0hYQpUY5xHNQzMJKDSfV33TrfnP1v9B2dJ2dpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSntYHR0nUvyTUDWJ4166icUs4UbK4w6ibYXgiaKL5zFGYo3RUkb85mSPng/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之间的未归一化转移概率， ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqS1ILNvLbFlZjsf5rnmNcHTQIvY8Zjkic7CoBpRhS6jictUeNHgEWIp0nQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是归一化常数。

node2vec引入两个超参数 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSbFb6d3GCyA1uut8jC6TEyYMjWrjy8C3VAyiaUZjRnArUkyKGWDSBiazQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqShLkEia99L2u6yehib3oiac0ErfXzRc5oINqD3tLSiavnEI7gmJMibR6MOag/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)来控制随机游走的策略，假设当前随机游走经过边 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSeMNuZVAlibOraVXuuGcGYPiaQsfvELgia5qOc5ianqF2a4Dr7oiaqo2wRBg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)到达顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSnkSxUJNKgMTaSibic8C8VKZiaDMmZ831ZN3khSo8PkPccXsThtTeDlz6w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)设![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSotbOBJqJ1K7qV9tWwNFsIv3s7q1zsc20zFnL5HTWXMnwmicdV6a4ibUA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSAvBpxFhRYfqAtrDu2xZdSVWfuyLdRB7Evf3YxyuHIu3icxBqOHTe3bg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSb38vItk2Dgcs4uPdaYRTHiaIXjTmcibgyzT4Le5860UY4bLAv3mjbPCA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqS8qaSRiasyMqICe4D3KQld6C5dYoyBhK1HDicXY0BVyW4tG024JrL6JXg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之间的边权，

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSXe8Cbe69gpjuFib7TcUyxHIClQ2QEACTs8LnqXAwFqEKibuVS7wvnrvA/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)



![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqS9VhatuWg1C4BcoQF5eh0AXk1mPCVAJQSeXf8wCFaGyN0loeAiaKnTFw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)为顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqS2PGP8eEe0ZqryCsyn6AbTCEiabl0qVRqdBqW9YR3dkmET6rveqib2IVA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和顶点![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSGHT3ArEAOUQDjSP8JuzKLNZs6L7jcyLC0hufJczPEMR8dysqFGEdag/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之间的最短路径距离。

下面讨论超参数p和 q对游走策略的影响



**参数p**控制重复访问刚刚访问过的顶点的概率。注意到p仅作用于![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSKkSRrxAibByLLWibxLlgvSAgZBVvuiaJqLkMseFXFRC7KHIZxECLRowicg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的情况，而 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSaxqDbFqXViaiaPTibEAqDHYrM7A3eZatnUh1rwZyeKc3t39rFxttU3Jog/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)表示顶点x就是访问当前顶点 ![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSw5R6gksq5fn3HyOcH7uCZHiasw3Vf626TD0PQl4O6EwaicpeOFd0qGmQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)之前刚刚访问过的顶点。那么若 p较高，则访问刚刚访问过的顶点的概率会变低，反之变高。



q控制着游走是向外还是向内，若![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSpMnv7OwSh7iazibIx1hDROJYuq9ibbNybRnorIWA0Cicwv1FSYxg3ocuOw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，随机游走倾向于访问和t接近的顶点(偏向BFS)。若![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSRPhdvS7MdQbdU9Hxd2lWAPrsAlHKEmP3tia0yaJDwiaibyVOq4l6Aaj0A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，倾向于访问远离t的顶点(偏向DFS)。

下面的图描述的是当从t访问到![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSb2Xb3bvNfbvYVLnPic3Tsicib1diaLyl97DqbVpDkzk7qdFXjdlF406gQw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)时，决定下一个访问顶点时每个顶点对应的![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSz07wNAm2Q3n0VBP1kZBl9QNPVbrtoogiaN1ZbcKYzSSXoPMuCjTMq7A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqSWC1ABko43gyHERRty15ytibBPwr1YvPgH61T0fDdc6bhWhzSYny0MLQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)



采样完顶点序列后，剩下的步骤就和deepwalk一样了，用word2vec去学习顶点的embedding向量。值得注意的是node2vecWalk中不再是随机抽取邻接点，而是按概率抽取，node2vec采用了Alias算法进行顶点采样。

**Alias Method:时间复杂度O(1)的离散采样方法**

https://zhuanlan.zhihu.com/p/54867139



**node2vec 核心代码**

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/zHbzQPKIBPhfOtVxRRmzCzW9RnzUQZqS7SNjia3yoIQKM3pEfwT2ZtXlrApiaUxSZALhD34wkHWCbzLTMDqQfAmw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)



通过上面的伪代码可以看到，node2vec和deepwalk非常类似，主要区别在于顶点序列的采样策略不同，所以这里我们主要关注node2vecWalk的实现。

**由于采样时需要考虑前面2步访问过的顶点，所以当访问序列中只有1个顶点时，直接使用当前顶点和邻居顶点之间的边权作为采样依据。当序列多余2个顶点时，使用文章提到的有偏采样**





**node2vec 应用**

使用node2vec在wiki数据集上进行节点分类任务和可视化任务。wiki数据集包含 2,405 个网页和17,981条网页之间的链接关系，以及每个网页的所属类别。通过简单的超参搜索，这里使用p=0.25,q=4的设置。







