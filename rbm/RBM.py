from utils import *

class RBM:
    def __init__(self, data_num = 2, m=10, n=5, k = 100, learning_rate = 1e-3):
        '''
            m : number of visible nodes
            n : number of hidden nodes
        '''
        self.visible_node = m
        self.hidden_node = n
        self.k = k
        self.data_num = data_num
        self.learning_rate= learning_rate
        
        self.W = tf.Variable(initialize_variable([self.hidden_node, self.visible_node], Type = 'uniform'), name = 'weights')
        self.b = tf.Variable(initialize_variable([self.visible_node], Type = 'uniform'), name = 'visible_biases')
        self.c = tf.Variable(initialize_variable([self.hidden_node], Type = 'uniform'), name = 'hidden_biases')
        
        self.visible = tf.placeholder(tf.float32, [self.data_num, self.visible_node])
        visible = iteration(self.visible, weights= self.W, hidden_biases=self.c, visible_biases=self.b, k=self.k)

        
        self.grad_W = tf.matmul(tf.transpose(linear(self.visible, weights = tf.transpose(self.W, [1,0]), biases = self.c), [1,0])
                                ,self.visible)-\
                    tf.matmul(tf.transpose(linear(visible, weights = tf.transpose(self.W, [1,0]), biases = self.c), [1,0])
                                ,visible)
        self.grad_b = tf.reduce_sum(self.visible-visible,[0])
        self.grad_c = tf.reduce_sum(linear(self.visible, weights = tf.transpose(self.W, [1,0]), biases = self.c)
                        -linear(visible, weights = tf.transpose(self.W, [1,0]), biases = self.c),[0])
        self.update_W = tf.assign(self.W, self.W + learning_rate*self.grad_W)
        self.update_b = tf.assign(self.b, self.b + learning_rate*self.grad_b)
        self.update_c = tf.assign(self.c, self.c + learning_rate*self.grad_c)
        self.update = [self.update_W, self.update_b, self.update_c]
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def visible_to_hidden(self, visible):
        return self.sess.run(tf.floor(linear(visible, weights = tf.transpose(self.W, [1,0]), biases = self.c)+0.5))    
    
    def train(self, train_data, train_steps = 100):
        self.sess.run(self.update, feed_dict = {self.visible : train_data})

    def print_tensors(self):
        print_tensor(self.sess, self.W)
        print_tensor(self.sess, self.b)
        print_tensor(self.sess, self.c)
    
    def get_free_energy(self, v, h):
        '''
            input
                v : 1D tensor m
                h : 1D tensor n
            return
                free energy
        '''
        if h.get_shape().ndims!=1 or v.get_shape().ndims!=1:
            raise ValueError("Dimension should be 1 but dimension h : {} and v : {}"
                                .format(h.get_shape().ndims, v.get_shape().ndims))
            
        if h.get_shape()[0]!=self.W.get_shape()[0] or v.get_shape()[0]!=self.W.get_shape()[1]:
            raise ValueError("Size note matches with variables")
        
        E1 = tf.matmul(tf.reshape(h, [1, -1]),self.W)
        E1 = tf.matmul(E1,tf.reshape(v, [-1, 1]))
        E1 = tf.reshape(E1, [1])
        E2 = tf.reshape(tf.matmul(tf.reshape(self.b, [1, -1]), tf.reshape(v,[-1, 1])), [1])
        E3 = tf.reshape(tf.matmul(tf.reshape(self.c, [1, -1]), tf.reshape(h,[-1, 1])), [1])
        energy = -E1-E2-E3
        print_tensor(self.sess, -E1-E2-E3)
        return energy

if __name__ == '__main__':
    r = RBM(data_num = 4, m = 3, n = 2)
    r.print_tensors()
    a = [[1,0,1],[0,1,0],[0,0,0],[1, 1, 1]]
    a = np.array(a,dtype = np.float32)
    r.train(a, train_steps= 10000000)
    r.print_tensors()
    print(r.visible_to_hidden(a))
