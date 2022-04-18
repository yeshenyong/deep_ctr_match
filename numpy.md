# numpy





### asarray

​	array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，属于深拷贝。但asarray不会，属于浅拷贝。、

```python
>>> a=[(1,2,3),(4,5,6)]
>>> a=np.asarray(a)
>>> print(a)
[[1 2 3]
 [4 5 6]]
>>> b=[(1,2,3),(4,5)]
>>> b=np.asarray(b)
>>> b.shape
(2,)
>>> print(b)
[(1, 2, 3) (4, 5)]
>>> print(a.shape)
(2, 3)
>>> c=[(1,2,3),(4,5,6)]
>>> c=np.asarray(c)
>>> c.shape
(2, 3)
```





### random



#### permutation

If `x` is a multi-dimensional array, it is only shuffled along its
            first index.

```python
Returns
    -------
    out : ndarray
    Permuted sequence or array range.

    Examples
    --------
    >>> np.random.permutation(10)
    array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

    >>> np.random.permutation([1, 4, 9, 12, 15])
    array([15,  1,  9,  4, 12])

    >>> arr = np.arange(9).reshape((3, 3))
    >>> np.random.permutation(arr)
    array([[6, 7, 8],
    [0, 1, 2],
    [3, 4, 5]])
```













