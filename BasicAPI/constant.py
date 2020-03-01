# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 16:38
# @Title:  : 

import tensorflow as tf

t = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])

print(t)

# tf.Tensor(
# [[1 2 3]
#  [4 5 6]], shape=(2, 3), dtype=int32)

print(t + 1)

# tf.Tensor(
# [[2 3 4]
#  [5 6 7]], shape=(2, 3), dtype=int32)

print(t @ tf.transpose(t))

# tf.Tensor(
# [[14 32]
#  [32 77]], shape=(2, 2), dtype=int32)

print(t.numpy())

# [[1 2 3]
#  [4 5 6]]

t = tf.constant('tensor')

print(t)
# tf.Tensor(b'tensor', shape=(), dtype=string)

print(tf.strings.unicode_decode(t, 'utf8'))
# tf.Tensor([116 101 110 115 111 114], shape=(6,), dtype=int32)

t = tf.constant(['tensor', 'flow'])

print(t)
# tf.Tensor([b'tensor' b'flow'], shape=(2,), dtype=string)

print(tf.strings.unicode_decode(t, 'utf8'))
# <tf.RaggedTensor [[116, 101, 110, 115, 111, 114], [102, 108, 111, 119]]>

r = tf.ragged.constant([
    [1, 2, 3],
    [4, 5, 6]
])

print(r)
# <tf.RaggedTensor [[1, 2, 3], [4, 5, 6]]>

print(r.to_tensor())

# tf.Tensor(
# [[1 2 3]
#  [4 5 6]], shape=(2, 3), dtype=int32)

s = tf.SparseTensor(
    indices=[[0, 0], [1, 2]],
    values=[1, 3.14],
    dense_shape=[3, 4]
)

print(s)

# SparseTensor(indices=tf.Tensor(
# [[0 0]
#  [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1.   3.14], shape=(2,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
