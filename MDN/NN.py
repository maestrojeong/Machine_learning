from utils import *

class Model_MSE:
    def __init__(self, num_data, epoch, batch_size, hidden_layers):
        self.num_data = num_data
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.model()

    def model(self):
        self.x_data = tf.placeholder(tf.float32, [None])
        self.y_data = tf.placeholder(tf.float32, [None])
        self.y_hat = Neural_network(self.x_data, 
                               hidden1_layers = self.hidden_layers,
                               hidden2_layers = self.hidden_layers
                               )
        self.error = tf.reduce_mean(tf.square(self.y_data - self.y_hat))
        self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.error)

    def run(self, train_data):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for _ in range(self.epoch):
            shuffle_data = shuffle(train_data)
            for batch in range(int(len(train_data['x'])/self.batch_size)): 
                sess.run(self.train_step, feed_dict={
                                            self.x_data : shuffle_data['x'][batch*self.batch_size:(batch+1)*self.batch_size], 
					    self.y_data : shuffle_data['y'][batch*self.batch_size:(batch+1)*self.batch_size]
                                            })
            if _%50==49:
                print("cost = {}".format(sess.run(self.error, feed_dict ={self.x_data : train_data['x'], self.y_data : train_data['y']})))
                y_temp = sess.run(self.y_hat, feed_dict = {self.x_data : train_data['x']})
                plt.plot(train_data['x'], train_data['y'], 'go')
                plotter(train_data['x'], y_temp, 'bo')        
        return y_temp
    

class Model_MDN:
    def __init__(self, num_data, epoch, batch_size, hidden_layers, K = 3, epsilon = 1e-4):
        self.num_data = num_data
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.K = K
        self.epsilon = epsilon
        self.model()

    def model(self):
        self.x_data = tf.placeholder(tf.float32, [None])
        self.y_data = tf.placeholder(tf.float32, [None])
        output_size = 3*self.K
        temp_x = tf.reshape(self.x_data, [-1,1])
        h1 = tf.tanh(fully_connected(temp_x, 1, self.hidden_layers))
        self.y_hat = tf.tanh(fully_connected(h1, self.hidden_layers, output_size))
        temp = tf.split(self.y_hat, [self.K, self.K, self.K], axis = 1 )
        self.pi = tf.nn.softmax(temp[0], dim = -1)
        self.sigma = tf.exp(temp[1])
        self.mu = temp[2]
        temp_y = stack_1D_to_2D(self.y_data, self.K)
        self.error = -tf.reduce_mean(tf.log(tf.reduce_sum(self.pi*tf.exp(tf.square(temp_y - self.mu)/(self.sigma+self.epsilon))/tf.sqrt(self.sigma+self.epsilon), [1])),[0])
        #print(tf.log(tf.reduce_mean(self.pi*tf.exp(tf.square(temp_y - self.mu)/self.sigma)/tf.sqrt(self.sigma), [1])))
        #print(self.error)
        self.train_step = tf.train.AdamOptimizer().minimize(self.error)

    def run(self, train_data):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for _ in range(self.epoch):
            shuffle_data = shuffle(train_data)
            for batch in range(int(len(train_data['x'])/self.batch_size)): 
                sess.run(self.train_step, feed_dict={
                                            self.x_data : shuffle_data['x'][batch*self.batch_size:(batch+1)*self.batch_size], self.y_data : shuffle_data['y'][batch*self.batch_size:(batch+1)*self.batch_size]
                                            })
            if _%50==49:
                print("cost = {}".format(sess.run(self.error, feed_dict ={self.x_data : train_data['x'], self.y_data : train_data['y']})))
                #y_temp = sess.run(self.y_hat, feed_dict = {self.x_data : train_data['x']})
                #plt.plot(train_data['x'], train_data['y'], 'go')
                #plotter(train_data['x'], y_temp, 'bo')        
        return {'pi' : sess.run(self.pi, feed_dict = {self.x_data : train_data['x']}),
                'sigma' : sess.run(self.sigma, feed_dict = {self.x_data : train_data['x']}),
                'mu' : sess.run(self.mu, feed_dict = {self.x_data : train_data['x']})
                }

