# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 20:43
# @Title:  : 


import tensorflow as tf
import os

file_names = []
for file in os.listdir('csv'):
    file_names.append('csv/' + file)

print(file_names)
# ['csv/AQI.csv', 'csv/CO.csv', 'csv/NO2.csv', 'csv/O3.csv', 'csv/PM10.csv', 'csv/PM2.5.csv', 'csv/SO2.csv']

file_dataset = tf.data.Dataset.list_files(file_names)

for file in file_dataset:
    print(file)

# tf.Tensor(b'csv\\PM10.csv', shape=(), dtype=string)
# tf.Tensor(b'csv\\NO2.csv', shape=(), dtype=string)
# tf.Tensor(b'csv\\CO.csv', shape=(), dtype=string)
# tf.Tensor(b'csv\\SO2.csv', shape=(), dtype=string)
# tf.Tensor(b'csv\\AQI.csv', shape=(), dtype=string)
# tf.Tensor(b'csv\\O3.csv', shape=(), dtype=string)
# tf.Tensor(b'csv\\PM2.5.csv', shape=(), dtype=string)

dataset = file_dataset.interleave(
    lambda filename : tf.data.TextLineDataset(filename),
    cycle_length=5
)

for line in dataset.take(8):
    print(line.numpy())