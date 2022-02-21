# collections 库



### namedtuple

**`namedtuple`(具名元组)**

```python
collections.namedtuple(typename, field_names, verbose=False, rename=False) 
```

返回一个具名元组子类 typename，其中参数的意义如下：

- **typename**：元组名称
- **field_names**: 元组中元素的名称
- **rename**: 如果元素名称中含有 python 的关键字，则必须设置为 rename=True
- **verbose**: 默认就好

下面来看看声明一个具名元组及其实例化的方法：

```python
import collections

# 两种方法来给 namedtuple 定义方法名
#User = collections.namedtuple('User', ['name', 'age', 'id'])
User = collections.namedtuple('User', 'name age id')
user = User('tester', '22', '464643123')

print(user)
```











