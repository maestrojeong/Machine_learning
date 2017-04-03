import tensorflow as tf
import numpy as np

def linear(x, weights, biases, activation = 'sigmoid'):
    '''
        input:
            x - 1D tensor
            fan_in - num of input nodes
            fan_out - num of output nodes
            activation - select the activation function
        return :
            1D tensor
            linear regression according to activation function 
    '''
    temp_linear = tf.matmul(x, weights) + biases
    if activation == 'sigmoid':
        return tf.sigmoid(temp_linear) 
    else : 
        return tf.nn.relu(temp_linear)

def sample(prob):
    '''
        input :
            prob 2D tensor
        return:

    '''
    return (tf.sign(prob - tf.random_uniform(prob.get_shape(),minval = 0, maxval=1, dtype = tf.float32)) + 1)/2

def iteration(x, weights, hidden_biases, visible_biases, k=3):
    temp = x
    for repeat in range(k):
        temp = iteration_unit(temp, weights = weights, hidden_biases = hidden_biases, visible_biases = visible_biases)
    return temp 

def iteration_unit(x, weights, hidden_biases, visible_biases):
    prob = linear(x, weights = tf.transpose(weights, [1,0]), biases = hidden_biases)
    prob2 = linear(sample(prob), weights = weights, biases = visible_biases)
    return sample(prob2)

def initialize_variable(shape, Type = 'uniform', min_val = -1.0, max_val = 1.0, mean = 0.0, sd = 0.1 ):
    if Type is 'uniform':
        return tf.random_uniform(shape, minval = min_val, maxval=max_val, dtype = tf.float32)
    

def print_tensor(sess, var):
    print(var.name)
    print(var.get_shape())
    print(sess.run(var))
