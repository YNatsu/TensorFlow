# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 16:54
# @Title:  : 

import tensorflow as tf

v = tf.Variable([
    [2., 3],
    [4, 5]
])

print(v)

# <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
# array([[2., 3.],
#        [4., 5.]], dtype=float32)>

print(v.value())

# tf.Tensor(
# [[2. 3.]
#  [4. 5.]], shape=(2, 2), dtype=float32)

print(v.numpy())

# [[2. 3.]
#  [4. 5.]]

v.assign(2*v)

print(v.numpy())
# [[ 4.  6.]
#  [ 8. 10.]]
