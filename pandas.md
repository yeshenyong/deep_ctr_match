# panda 库

### DataFrame

字典类，灵活使用多项功能，配合panda 内部其他函数进行使用

https://blog.csdn.net/wei_lin/article/details/93492252



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





