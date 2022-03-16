# panda 库



核心数据结构：

`Series` 和 `DataFrame`

分别代表着一维的序列和二维的表结构



### Series

定长的字典序列。说是定长是因为在存储的时候，相当于两个 ndarray，这也是和字典结构最大的不同。因为在字典的结构里，元素的个数是不固定的。

```python
import pandas as pd
from pandas import Series, DataFrame
x1 = Series([1,2,3,4])
x2 = Series(data=[1,2,3,4], index=['a', 'b', 'c', 'd'])
print x1
print x2
```



上面这个例子中，x1 中的 index 采用的是默认值，x2 中 index 进行了指定。我们也可以采用字典的方式来创建 Series，比如：

```Python
d = {'a':1, 'b':2, 'c':3, 'd':4}
x3 = Series(d)
print x3 
```



#### **Series的增删改查**

1. 创建一个Series

```python
In [85]: ps = pd.Series(data=[-3,2,1],index=['a','f','b'],dtype=np.float32)     

In [86]: ps                                                                     
Out[86]: 
a   -3.0
f    2.0
b    1.0
dtype: float32
```



2. 增加元素append

```python
In [112]: ps.append(pd.Series(data=[-8.0],index=['f']))                         
Out[112]: 
a    4.0
f    2.0
b    1.0
f   -8.0
dtype: float64
```



3. 删除元素drop

```python
In [119]: ps                                                                    
Out[119]: 
a    4.0
f    2.0
b    1.0
dtype: float32

In [120]: psd = ps.drop('f')                                                    
In [121]: psd                                                                  
Out[121]: 
a    4.0
b    1.0
dtype: float32
```

注意不管是 append 操作，还是 drop 操作，都是发生在原数据的副本上，不是原数据上。



4. 修改元素

```python
In [123]: psn                                                                   
Out[123]: 
a    4.0
f    2.0
b    1.0
f   -8.0
dtype: float64

In [124]: psn['f'] = 10.0                                                       
In [125]: psn                                                                   
Out[125]: 
a     4.0
f    10.0
b     1.0
f    10.0
dtype: float64
```



5. 访问元素

一种通过默认的整数索引，在 Series 对象未被显示的指定 label 时，都是通过索引访问；另一种方式是通过标签访问。

```python
In [126]: ps                                                                    
Out[126]: 
a    4.0
f    2.0
b    1.0
dtype: float32

In [128]: ps[2] # 索引访问                              
Out[128]: 1.0

In [127]: ps['b']  # 标签访问                                                             
Out[127]: 1.0
```







### DataFrame

**DataFrame 类型数据结构类似数据库表**。它包括了行索引和列索引，我们可以将 DataFrame 看成是由相同索引的 Series 组成的字典类型。

字典类，灵活使用多项功能，配合panda 内部其他函数进行使用

https://blog.csdn.net/wei_lin/article/details/93492252



```python
import pandas as pd
from pandas import Series, DataFrame
data = {'Chinese': [66, 95, 93, 90,80],'English': [65, 85, 92, 88, 90],'Math': [30, 98, 96, 77, 90]}
df1= DataFrame(data)
df2 = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'], columns=['English', 'Math', 'Chinese'])
print(df1)
print(df2)
```

​	在后面的案例中，我一般会用 df, df1, df2 这些作为 DataFrame 数据类型的变量名，我们以例子中的 df2 为例，列索引是[‘English’, ‘Math’, ‘Chinese’]，行索引是[‘ZhangFei’, ‘GuanYu’, ‘ZhaoYun’, ‘HuangZhong’, ‘DianWei’]，所以 df2 的输出是：

```python
            English  Math  Chinese
ZhangFei         65    30       66
GuanYu           85    98       95
ZhaoYun          92    96       93
HuangZhong       88    77       90
DianWei          90    90       80
```



#### 数据的导入与输出

​	Pandas 允许直接从 xlsx，csv 等文件中导入数据，也可以输出到 xlsx, csv 等文件，非常方便。

​	需要说明的是，在运行的过程可能会存在缺少 xlrd 和 openpyxl 包的情况，到时候如果缺少了，可以在命令行模式下使用“pip install”命令来进行安装。

```python
import pandas as pd
from pandas import Series, DataFrame
score = DataFrame(pd.read_excel('data.xlsx'))
score.to_excel('data1.xlsx')
print(score)
```

关于数据导入， pandas提供了强劲的读取支持， 比如读写CSV文件， `read_csv()`函数有38个参数之多， 这里面有一些很有用， 主要可以分为下面几个维度来梳理：

- [https://mp.weixin.qq.com/s/5KSjitB6LRpRsEnbhs1xcg]()



#### 数据清洗

​	数据清洗是数据准备过程中必不可少的环节，Pandas 也为我们提供了数据清洗的工具，在后面数据清洗的章节中会给你做详细的介绍，这里简单介绍下 Pandas 在数据清洗中的使用方法。

**删除 DataFrame 中的不必要的列或行**

Pandas 提供了一个便捷的方法 drop() 函数来删除我们不想要的列或行

```python
df2 = df2.drop(columns=['Chinese'])
```

想把“张飞”这行删掉。

```python
df2 = df2.drop(index=['ZhangFei'])
```



> 如果你想对 DataFrame 中的 columns 进行重命名，可以直接使用 rename(columns=new_names, inplace=True) 函数，比如我把列名 Chinese 改成 YuWen，English 改成 YingYu。

```python
df2.rename(columns={'Chinese': 'YuWen', 'English': 'Yingyu'}, inplace = True)
```



**去重复的值**

数据采集可能存在重复的行，这时只要使用 drop_duplicates() 就会自动把重复的行去掉

```python
df = df.drop_duplicates() #去除重复行
```



**格式问题**

- 更改数据格式

​	这是个比较常用的操作，因为很多时候数据格式不规范，我们可以使用 astype 函数来规范数据格式，比如我们把 Chinese 字段的值改成 str 类型，或者 int64 可以这么写

```python
df2['Chinese'].astype('str') 
df2['Chinese'].astype(np.int64) 
```



- 数据间的空格

​	有时候我们先把格式转成了 str 类型，是为了方便对数据进行操作，这时想要删除数据间的空格，我们就可以使用 strip 函数：

```python
#删除左右两边空格
df2['Chinese']=df2['Chinese'].map(str.strip)
#删除左边空格
df2['Chinese']=df2['Chinese'].map(str.lstrip)
#删除右边空格
df2['Chinese']=df2['Chinese'].map(str.rstrip)
```



如果数据里有某个特殊的符号，我们想要删除怎么办？同样可以使用 strip 函数，比如 Chinese 字段里有美元符号，我们想把这个删掉，可以这么写：

```python
df2['Chinese']=df2['Chinese'].str.strip('$')
```



- 大小写转换

大小写是个比较常见的操作，比如人名、城市名等的统一都可能用到大小写的转换，在 Python 里直接使用 upper(), lower(), title() 函数，方法如下：

```python
#全部大写
df2.columns = df2.columns.str.upper()
#全部小写
df2.columns = df2.columns.str.lower()
#首字母大写
df2.columns = df2.columns.str.title()
```



- 查找空值

数据量大的情况下，有些字段存在空值 NaN 的可能，这时就需要使用 Pandas 中的 isnull 函数进行查找。比如，我们输入一个数据表如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsKwIA7etrrjeVPgcEz6nvlWoib9pxpAzBldRorEk2yd0TRm6fuBUEwibA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



如果我们想看下哪个地方存在空值 NaN，可以针对数据表 df 进行 df.isnull()，结果如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsI1bebicTptNjk9JJXSfMqADSCncAq6rAKcPewlKmWF8jG0S8tmZBr3Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

如果我想知道哪列存在空值，可以使用 df.isnull().any()，结果如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFs4Nh1biaymmlsK9VozVY8frjaBPQ94AF7WxeNSTsZrRhL7ApwtkadDyQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



**使用apply函数对数据进行清洗**

apply 函数是 Pandas 中自由度非常高的函数，使用频率也非常高。比如我们想对 name 列的数值都进行大写转化可以用：

```python
df['name'] = df['name'].apply(str.upper)
```

​	我们也可以定义个函数，在 apply 中进行使用。比如定义 double_df 函数是将原来的数值 *2 进行返回。然后对 df1 中的“语文”列的数值进行 *2 处理，可以写成：

```python
def double_df(x):
           return 2*x
df1[u'语文'] = df1[u'语文'].apply(double_df)
```



​	我们也可以定义更复杂的函数，比如对于 DataFrame，我们新增两列，其中’new1’列是“语文”和“英语”成绩之和的 m 倍，'new2’列是“语文”和“英语”成绩之和的 n 倍，我们可以这样写：

```python
def plus(df,n,m):
    df['new1'] = (df[u'语文']+df[u'英语']) * m
    df['new2'] = (df[u'语文']+df[u'英语']) * n
    return df
df1 = df1.apply(plus,axis=1,args=(2,3,))
```



#### 数据统计

​	在数据清洗后，我们就要对数据进行统计了。Pandas 和 NumPy 一样，都有常用的统计函数，如果遇到空值 NaN，会自动排除。常用的统计函数包括：

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsV5OuPdUuQdWmloFl2PKYF5gv0xuNALS90iaeEE8XytSgW9zr47ZBBDg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

​	表格中有一个 describe() 函数，统计函数千千万，describe() 函数最简便。它是个统计大礼包，可以快速让我们对数据有个全面的了解。下面我直接使用 df1.descirbe() 输出结果为：

```python
df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
print df1.describe()
```

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsuAm5A23iaOBlsDmribmoyvylU3gMUde6tnRTYmz82lbFlqGzAN5wEbCw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

#### 数据表合并

有时候我们需要将多个渠道源的多个数据表进行合并，一个 DataFrame 相当于一个数据库的数据表，那么多个 DataFrame 数据表的合并就相当于多个数据库的表合并。
比如我要创建两个 DataFrame：

```python
df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
df2 = DataFrame({'name':['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2':range(5)})
```

两个 DataFrame 数据表的合并使用的是 merge() 函数，有下面 5 种形式：

1. 基于指定列进行连接

比如我们可以基于 name 这列进行连接。

```python
df3 = pd.merge(df1, df2, on='name')
```



2. inner内连接

inner 内链接是 merge 合并的默认情况，inner 内连接其实也就是键的交集，在这里 df1, df2 相同的键是 name，所以是基于 name 字段做的连接：

```python
df3 = pd.merge(df1, df2, how='inner')
```

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsfKbrUibfAvHYkPxTCwiaOib3BE1VDibI2Fhg4DdH6XWF7Wic2QHyVS0ZBoQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)







3. left左连接

左连接是以第一个 DataFrame 为主进行的连接，第二个 DataFrame 作为补充。

4. right右连接

右连接是以第二个 DataFrame 为主进行的连接，第一个 DataFrame 作为补充。

5. outer外连接

外连接相当于求两个 DataFrame 的并集。

#### DataFram的行级遍历

​	尽管 Pandas 已经尽可能向量化，让使用者尽可能避免 for 循环，但是有时不得已，还得要遍历 DataFrame。Pandas 提供 iterrows、itertuples 两种行级遍历。

1. 使用 `iterrows` 遍历打印所有行，在 IPython 里输入以下行：

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsSiclA3ibMWeUicGL4tJSgUCiaSgWvLYicFppcaS5DJ09bjUiawTMAlF5HvgQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



2. 使用 `itertuples` 遍历打印每行：这个效率更高， 比上面那个节省6倍多的时间， 所在数据量非常大的时候， 推荐后者。访问每一行某个元素的时候， 需要`getattr`函数

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsZfYWOUvsLf2ra5icQOjGuqUAmQJAQI7fpuAvEMxhWQCc60ybMo3eib3A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



3. 使用`iteritems`遍历每一行

​	这个访问每一行元素的时候， 用的是每一列的数字索引

![图片](https://mmbiz.qpic.cn/mmbiz_png/vI9nYe94fsFbZh04AsPta2sVVzYf8YFsD8lKZHHB1nAHDEf10qdtaGAGuWvuMvXzHBfMoFrebQtuIv5HbEjLAQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)









`fillna` 函数

method

> Fill NA/NaN values using the specified method.

Examples

```python
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, np.nan],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))
df
     A    B   C    D
0  NaN  2.0 NaN  0.0
1  3.0  4.0 NaN  1.0
2  NaN  NaN NaN  NaN
3  NaN  3.0 NaN  4.0
```

Replace all NaN elements with 0s.

```python
>>> df.fillna(0)
     A    B    C    D
0  0.0  2.0  0.0  0.0
1  3.0  4.0  0.0  1.0
2  0.0  0.0  0.0  0.0
3  0.0  3.0  0.0  4.0
```

Replace all NaN elements in column ‘A’, ‘B’, ‘C’, and ‘D’, with 0, 1, 2, and 3 respectively.

```python
>>> values = {"A": 0, "B": 1, "C": 2, "D": 3}
>>> df.fillna(value=values)
     A    B    C    D
0  0.0  2.0  2.0  0.0
1  3.0  4.0  2.0  1.0
2  0.0  1.0  2.0  3.0
3  0.0  3.0  2.0  4.0
```

Only replace the first NaN element.

```python
>>> df.fillna(value=values, limit=1)
     A    B    C    D
0  0.0  2.0  2.0  0.0
1  3.0  4.0  NaN  1.0
2  NaN  1.0  NaN  3.0
3  NaN  3.0  NaN  4.0
```

When filling using a DataFrame, replacement happens along the same column names and same indices

```python
>>> df2 = pd.DataFrame(np.zeros((4, 4)), columns=list("ABCE"))
>>> df.fillna(df2)
     A    B    C    D
0  0.0  2.0  0.0  0.0
1  3.0  4.0  0.0  1.0
2  0.0  0.0  0.0  NaN
3  0.0  3.0  0.0  4.0
```





### Merge 函数

Merge介绍

- pandas中的merge()函数类似于SQL中join的用法，可以将不同数据集依照某些字段（属性）进行合并操作，得到一个新的数据集。

https://blog.csdn.net/sinat_31854967/article/details/108989202



功能介绍

- 构建两张表
- 两表合并
- 两个键合并
- 如果有不一样的直接merge会被过滤掉默认为交集
- 求并集
- 以左右表为基准
- join连表操作





### 如何用SQL方式打开Pandas

​	Pandas 的 DataFrame 数据类型可以让我们像处理数据表一样进行操作，比如数据表的增删改查，都可以用 Pandas 工具来完成。不过也会有很多人记不住这些 Pandas 的命令，相比之下还是用 SQL 语句更熟练，用 SQL 对数据表进行操作是最方便的，它的语句描述形式更接近我们的自然语言。

​	事实上，在 Python 里可以直接使用 SQL 语句来操作 Pandas。

​	这里给你介绍个工具：pandasql。



```python
import pandas as pd
from pandas import DataFrame
from pandasql import sqldf, load_meat, load_births
df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
pysqldf = lambda sql: sqldf(sql, globals())
sql = "select * from df1 where name ='ZhangFei'"
print pysqldf(sql)
```



在这个例子里，输入的参数是 sql，返回的结果是 sqldf 对 sql 的运行结果，当然 sqldf 中也输入了 globals 全局参数，因为在 sql 中有对全局参数 df1 的使用。

