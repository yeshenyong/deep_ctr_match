

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







### 默认

#### build

#### call

