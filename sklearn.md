# sklearn 库

### sklearn.metrics

`log_loss`

−(ylog(p)+(1−y)log(1−p))

y yy表示样本的真实标签(1或0)，p pp表示模型预测为正样本的概率。



所以最差的情况就是，正好是一半正样本一半负样本，此时你乱猜出的logloss是0.693。

所以只要loglss是在0.693以上，就说明模型是失败的。



`roc_auc_score`

```python
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
```

计算 auc 指标





### sklearn.model_selection

`train_test_split`







### sklearn.preprocessing

`LabelEncoder`

将类别数字化

直接看官方文档（直观明了）

Example

```python
>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
>>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
array([0, 0, 1, 2]...)
>>> le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])

It can also be used to transform non-numerical labels (as long as they are
                                                       hashable and comparable) to numerical labels.

>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
```



`LabelEncoder.fit_transform`

即融合 `fit` 和 `transform`





`MinMaxScaler`

数据归一化操作

```python
# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
print(sc)

MinMaxScaler(copy=True, feature_range=(0, 1))

-------------

training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
print(training_set_scaled[:5,])
print(test_set[:5,])

[[0.011711  ]
 [0.00980951]
 [0.00540518]
 [0.00590914]
 [0.00489135]]
[[0.84288404]
 [0.85345726]
 [0.84641315]
 [0.87046756]
 [0.86758781]]
```

`MinMaxScaler.fit_transform`

即融合 `fit` 和 `transform`





`MultiLabelBinarizer`

多类别或多标签分类问题，有两种构建[分类器](https://so.csdn.net/so/search?q=分类器&spm=1001.2101.3001.7020)的策略：One-vs-All及One-vs-One。

对于多标签分类问题而言，一个样本可能同时属于多个类别。如一个新闻属于多个话题。这种情况下，因变量yy需要使用一个[矩阵](https://so.csdn.net/so/search?q=矩阵&spm=1001.2101.3001.7020)表达出来。

example

```python
#
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2,3,4],[2],[0,1,3],[0,1,2,3,4],[0,1,2]]
MultiLabelBinarizer().fit_transform(y) #这里没指名classes 则classes从训练数据获取这里为         
                                       #[0,1,2,3,4]
# array([[0, 0, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0]])
 
In [3]: from sklearn.preprocessing import MultiLabelBinarizer
   ...: mlb = MultiLabelBinarizer(classes = [2,3,4,5,6,1])
   ...: mlb.fit_transform([(1, 2), (3,4),(5,)])
   ...:
Out[3]:
array([[1, 0, 0, 0, 0, 1],
       [0, 1, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0]])
 
In [4]: mlb.classes_
Out[4]: array([2, 3, 4, 5, 6, 1])
```





### sklearn.multiclass

`OneVsRestClassifier`、`OneVsOneClassifier`

​	针对多类问题的分类中，具体讲有两种，即multiclass classification和multilabel classification。multiclass是指分类任务中包含不止一个类别时，每条数据仅仅对应其中一个类别，不会对应多个类别。multilabel是指分类任务中不止一个分类时，每条数据可能对应不止一个类别标签，例如一条新闻，可以被划分到多个板块。
 无论是multiclass，还是multilabel，做分类时都有两种策略，一个是one-vs-the-rest(one-vs-all)，一个是one-vs-one。

​	在one-vs-all策略中，假设有n个类别，那么就会建立n个二项分类器，每个分类器针对其中一个类别和剩余类别进行分类。进行预测时，利用这n个二项分类器进行分类，得到数据属于当前类的概率，选择其中概率最大的一个类别作为最终的预测结果。

​	在one-vs-one策略中，同样假设有n个类别，则会针对两两类别建立二项分类器，得到k=n*(n-1)/2个分类器。对新数据进行分类时，依次使用这k个分类器进行分类，每次分类相当于一次投票，分类结果是哪个就相当于对哪个类投了一票。在使用全部k个分类器进行分类后，相当于进行了k次投票，选择得票最多的那个类作为最终分类结果。

​	在scikit-learn框架中，分别有sklearn.multiclass.OneVsRestClassifier和sklearn.multiclass.OneVsOneClassifier完成两种策略，使用过程中要指明使用的二项分类器是什么。另外在进行mutillabel分类时，训练数据的类别标签Y应该是一个矩阵，第[i,j]个元素指明了第j个类别标签是否出现在第i个样本数据中。例如，np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])，这样的一条数据，指明针对第一条样本数据，类别标签是第0个类，第二条数据，类别标签是第1，第2个类，第三条数据，没有类别标签。有时训练数据中，类别标签Y可能不是这样的可是，而是类似[[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]这样的格式，每条数据指明了每条样本数据对应的类标号。这就需要将Y转换成矩阵的形式，sklearn.preprocessing.MultiLabelBinarizer提供了这个功能。







### sklearn.manifold

`TSNE`

```python
# 高维embedding 映射到 2D space
model = TSNE(n_components=2)
node_pos = model.fit_transform(emb_list)

for c, idx in color_idx.items():
    plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
```





