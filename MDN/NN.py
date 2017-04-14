from utils import *

class Model_MSE:
    def __init__(self, epoch, batch_size, hidden_layers):
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.sess = tf.Session()
        self.model()
    
    def model(self):
        self.x_data = tf.placeholder(tf.float32, [None])
        self.y_data = tf.placeholder(tf.float32, [None])
        self.y_hat = Neural_network(self.x_data, hidden_layers = self.hidden_layers)
        self.error = tf.reduce_mean(tf.square(self.y_data - self.y_hat))
        self.global_step = tf.Variable(0, trainable = False) 
        learning_rate = tf.train.exponential_decay(learning_rate = 1e-1, global_step = self.global_step, decay_steps = 100, decay_rate = 0.9, staircase = True)   
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss = self.error, global_step = self.global_step)

    def test(self, test_data):
        y_temp = self.sess.run(self.y_hat, feed_dict = {self.x_data : test_data['x']})
        plt.plot(test_data['x'], test_data['y'], 'bo', alpha = 0.3)
        plotter(test_data['x'], y_temp, 'ro', alpha = 0.3)        
    
    def reset(self):
        self.sess.run(tf.assign(self.global_step, 0))

    def run(self, train_data, plot = True):
        self.sess.run(tf.global_variables_initializer())
        for _ in range(self.epoch):
            shuffle_data = shuffle(train_data)
            for batch in range(int(len(train_data['x'])/self.batch_size)): 
                self.sess.run(self.train_step, feed_dict={
                                            self.x_data : shuffle_data['x'][batch*self.batch_size:(batch+1)*self.batch_size], 
					    self.y_data : shuffle_data['y'][batch*self.batch_size:(batch+1)*self.batch_size]
                                            })
            if _%200==199:
                print("cost = {}".format(self.sess.run(self.error, feed_dict ={self.x_data : train_data['x'], self.y_data : train_data['y']})))
                if plot == True:
                    y_temp = self.sess.run(self.y_hat, feed_dict = {self.x_data : train_data['x']})
                    plt.plot(train_data['x'], train_data['y'], 'bo', alpha = 0.3)
                    plotter(train_data['x'], y_temp, 'ro', alpha = 0.3)        

class Model_MDN:
    def __init__(self, epoch, batch_size, hidden_layers, K = 3, epsilon = 1e-4):
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.K = K
        self.epsilon = epsilon
        self.sess = tf.Session()
        self.model()
    
    def graph(self, test_data, k):
        if k>=self.K: 
            print("{} should be lower than {}".format(k, self.K))
            return
        pi = self.sess.run(self.pi, feed_dict = {self.x_data : test_data['x']})
        sigma = self.sess.run(self.sigma, feed_dict = {self.x_data : test_data['x']})
        mu = self.sess.run(self.mu, feed_dict = {self.x_data : test_data['x']})
        plt.title("{}, {}".format('pi', k+1))
        plotter(test_data['x'], pi[:,k],'bo', alpha = 0.3)
        plt.title("{}, {}".format('sigma', k+1))
        plotter(test_data['x'], sigma[:,k],'bo', alpha = 0.3)
        plt.title("{}, {}".format('mu', k+1))
        plotter(test_data['x'], mu[:,k],'bo', alpha = 0.3)

    def test(self, test_data):
        pi = self.sess.run(self.pi, feed_dict = {self.x_data : test_data['x']})
        sigma = self.sess.run(self.sigma, feed_dict = {self.x_data : test_data['x']})
        mu = self.sess.run(self.mu, feed_dict = {self.x_data : test_data['x']})
        length = len(test_data['x'])
        y_temp = np.zeros(length)

        for i in range(length):
            index = random_choice(self.K, pi[i])
            temp_sigma = sigma[i][index]
            temp_mu = mu[i][index]
            y_temp[i] = temp_sigma*np.random.randn()+temp_mu
        
        plt.plot(test_data['x'], test_data['y'], 'bo', alpha = 0.3)
        plotter(test_data['x'], y_temp, 'ro', alpha = 0.3)

    def model(self):
        self.x_data = tf.placeholder(tf.float32, [None])
        self.y_data = tf.placeholder(tf.float32, [None])
        output_size = 3*self.K
        self.output = Neural_network(self.x_data, self.hidden_layers, output_size, stddev = 0.1)
        temp = tf.split(self.output, [self.K, self.K, self.K], axis = 1)
        self.pi = tf.nn.softmax(temp[0], dim = -1)
        self.sigma = tf.exp(temp[1])
        self.mu = temp[2]
        temp_y = stack_1D_to_2D(self.y_data, self.K)
        self.error = MDN_error(temp_y, self.pi, self.mu, self.sigma, self.epsilon)
        self.train_step = tf.train.AdamOptimizer().minimize(loss = self.error)

    def run(self, train_data):
        self.sess.run(tf.global_variables_initializer())
        for _ in range(self.epoch):
            shuffle_data = shuffle(train_data)
            for batch in range(int(len(train_data['x'])/self.batch_size)): 
                self.sess.run(self.train_step, feed_dict={
                                            self.x_data : shuffle_data['x'][batch*self.batch_size:(batch+1)*self.batch_size],
                                            self.y_data : shuffle_data['y'][batch*self.batch_size:(batch+1)*self.batch_size]
                                            })
            if _%100==99:    
                print("cost = {}".format(self.sess.run(self.error, feed_dict ={self.x_data : train_data['x'], self.y_data : train_data['y']})))
