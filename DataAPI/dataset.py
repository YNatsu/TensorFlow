# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 18:20
# @Title:  : 

import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices(np.arange(6))

for i in dataset:
    print(i)

dataset = dataset.repeat(3).batch(7)

for i in dataset:
    print(i)

# tf.Tensor([0 1 2 3 4 5 0], shape=(7,), dtype=int32)
# tf.Tensor([1 2 3 4 5 0 1], shape=(7,), dtype=int32)
# tf.Tensor([2 3 4 5], shape=(4,), dtype=int32)
