

# tensorflow.python.keras.layers

### Input

Input():用来实例化一个[keras](https://so.csdn.net/so/search?q=keras&spm=1001.2101.3001.7020)张量

```python
Input(shape=None,batch_shape=None,name=None,dtype=K.floatx(),sparse=False,tensor=None)
```



### Lambda



### preprocessing

#### sequence

##### pad_sequenes

```python
tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre',
    value=0.0
)
```

序列化数据填充

sequences：浮点数或整数构成的两层嵌套列表

maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.在命名实体识别任务中，主要是指句子的最大长度

dtype：返回的numpy array的数据类型

padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补

truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断

value：浮点数，此值将在填充时代替默认的填充值0

Returns

```python
x: Numpy array with shape `(len(sequences), maxlen)`
```



### layers

#### Dense

`Dense` implements the operation: `output = activation(dot(input, kernel) + bias)` where `activation` is the element-wise activation function passed as the `activation` argument, `kernel` is a weights matrix created by the layer, and `bias` is a bias vector created by the layer (only applicable if `use_bias` is `True`).

Example:

```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```



#### add

| Arguments  |                                       |
| :--------- | ------------------------------------- |
| `inputs`   | A list of input tensors (at least 2). |
| `**kwargs` | Standard layer keyword arguments.     |

| Returns                          |
| :------------------------------- |
| A tensor, the sum of the inputs. |



#### BatchNormalization



#### Activation

Applies an activation function to an output.



#### Embedding

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

Embedding层只能作为模型的第一层

参数

input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
output_dim：大于0的整数，代表全连接嵌入的维度
embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象
embeddings_constraint: 嵌入矩阵的约束项，为Constraints对象
mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 2。
input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断







### 默认

继承keras.layer

#### build



#### call

