

# tensorflow.python.keras.layers

### Input

Input():用来实例化一个[keras](https://so.csdn.net/so/search?q=keras&spm=1001.2101.3001.7020)张量

```python
Input(shape=None,batch_shape=None,name=None,dtype=K.floatx(),sparse=False,tensor=None)
```



### Lambda





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

