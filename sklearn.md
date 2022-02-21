# sklearn 库

### sklearn.metrics

`log_loss`





`roc_auc_score`







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





