from RBM import *

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

batch_size = 10
batch = mnist.train.next_batch(batch_size)

rbm_input = np.concatenate((to_binary_2D(batch[0]), batch[1]), axis = 1)
print(rbm_input.shape)

r = RBM(data_num = batch_size, m = 28*28+10, n = 20, k = 100)
r.print_tensors()

