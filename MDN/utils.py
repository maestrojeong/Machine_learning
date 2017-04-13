import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

def random_sample(length, Min, Max):
    '''return random seqeunce each value located between Min and Max'''
    return Min+(Max-Min)*np.random.rand(int(length))

def function(x, error_bound =0.1):
    length = len(x)
    target = np.zeros(length)
    for i in range(length):
        target[i] = x[i] + 0.3*math.sin(2*math.pi*x[i])
    target += random_sample(length, -error_bound, error_bound)
    return target

def plotter(x, y, option = 'b'):
    plt.plot(x, y , option)
    plt.show()

def fully_connected(input, in_layer, out_layer):
    W = tf.Variable(tf.truncated_normal([in_layer, out_layer], dtype = tf.float32)*1,name = 'fc_weight')
    b = tf.Variable(tf.random_uniform(shape=[out_layer], dtype = tf.float32)*0.01, name = 'fc_bias')
    return tf.matmul(input, W)+b

def Neural_network(x, hidden1_layers, hidden2_layers):
    temp_x = tf.reshape(x, [-1, 1])
    h1 = tf.tanh(fully_connected(temp_x, 1, hidden1_layers))
    h2 = tf.tanh(fully_connected(h1, hidden1_layers, hidden2_layers))
    y_hat = fully_connected(h2, hidden2_layers, 1)
    return tf.reshape(y_hat, [-1])

def stack_1D_to_2D(x, num):
    temp = tf.tile(x,[num])
    temp2 = tf.split(temp, num, axis = 0)
    temp3 = tf.stack(temp2)
    return tf.transpose(temp3, [1,0])

def shuffle(dataset):
    length = len(dataset['x'])
    shuffle = np.arange(0, length)
    np.random.shuffle(shuffle)
    shuffle_data = {}
    shuffle_data['x'] = []
    shuffle_data['y'] = []
    
    for i in range(length):
        shuffle_data['x'].append(dataset['x'][shuffle[i]])
        shuffle_data['y'].append(dataset['y'][shuffle[i]])
    
    shuffle_data['x'] = np.array(shuffle_data['x'])
    shuffle_data['y'] = np.array(shuffle_data['y'])
    return shuffle_data
