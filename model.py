import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class HebbLearner():
    def __init__(self,config):
        self.num_sequence = 1000
        self.config = config
        self.batch_size = 1
        self.lr = 0.001
        self.model = self.model()

    def create_dataset(self):
        xcenters = np.array([-1,1])
        ycenters = np.array([1,-1])

        labels = np.random.randint(2, size=self.num_sequence)

        data = np.zeros((self.num_sequence,2))

        mean = (xcenters[0],ycenters[0])
        cov = [[0.5, 0], [0, 0.5]]
        data[labels==0,:] = np.random.multivariate_normal(mean, cov, (np.sum(labels==0)))

        mean = (xcenters[1],ycenters[1])
        data[labels==1,:] = np.random.multivariate_normal(mean, cov, (np.sum(labels==1)))

        # whether to show the graph or not
        if self.config.show_graph:
            plt.scatter(data[labels==1,0], data[labels==1,1])
            plt.scatter(data[labels==0,0], data[labels==0,1])
            plt.show()

        return data, labels.astype(np.int32)

    def _weightVar(self,shape, mean=0.0, stddev=0.01, name='weights'):
        init_w = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        weights = tf.get_variable(name=name,initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=stddev))

        return weights

    def model(self):
        sequences = tf.placeholder(tf.float32, [None, 2])
        labels = tf.placeholder(tf.int32, [None])

        one_hot_labels = tf.one_hot(labels,2)

        """ forward pass """
        # simple archecture 2->32->32->2
        # activation: relu
        # first layer
        with tf.variable_scope('fw'):
            W_1 = self._weightVar([2,32],name='W1')
            y_1 = tf.nn.tanh(tf.matmul(sequences,W_1))

            # second layer
            W_2 = self._weightVar([32,32],name='W2')
            y_2= tf.nn.tanh(tf.matmul(y_1,W_2))

            # third layer
            W_3 = self._weightVar([32,1],name='W3')
            y_3 = tf.sign(tf.nn.relu(tf.matmul(y_2,W_3)))

            miss_list = tf.not_equal(tf.cast(tf.squeeze(y_3),tf.float64),tf.cast(labels,tf.float64))
            miss_rate = tf.reduce_sum(tf.cast(miss_list,tf.float32))/(self.batch_size)

        update_ops = []
        with tf.variable_scope('bw'):
            #[W_3.assign(W_3+self.lr*tf.matmul(tf.transpose(y_2),tf.cast(tf.expand_dims(labels,1),tf.float32)))])
            update_ops.extend([W_3.assign(W_3+self.lr*tf.matmul(tf.transpose(y_2),tf.cast(2*tf.expand_dims(labels,1)-1,tf.float32)))])
            update_ops.extend([W_2.assign(W_2+self.lr*tf.matmul(tf.transpose(y_1),y_2))])
            update_ops.extend([W_1.assign(W_1+self.lr*tf.matmul(tf.transpose(sequences),y_1))])
            with tf.control_dependencies([y_3]):
                backwards_op = tf.tuple(update_ops)

        return AttrDict(locals())

    def train(self,sess,data,labels):
        model = self.model
        for epoch_ind in range(5):
            miss_rate_list = []
            for bi in range(int(data.shape[0]/self.batch_size)):
                batch_data = data[bi*self.batch_size:(bi+1)*self.batch_size,:]
                batch_labels = labels[bi*self.batch_size:(bi+1)*self.batch_size]
                W,miss_rate = sess.run([model.backwards_op,model.miss_rate],feed_dict={model.sequences:batch_data,model.labels:batch_labels})
                miss_rate_list.append(miss_rate)

            print("Tranning Error at Epochs{}:{}".format(epoch_ind,np.mean(miss_rate_list)))

    def test(self,sess,data,labels):
        model = self.model
        miss_rate_list = []
        for bi in range(int(data.shape[0]/self.batch_size)):
            batch_data = data[bi*self.batch_size:(bi+1)*self.batch_size,:]
            batch_labels = labels[bi*self.batch_size:(bi+1)*self.batch_size]
            miss_rate = sess.run([model.miss_rate],feed_dict={model.sequences:batch_data,model.labels:batch_labels})
            miss_rate_list.append(miss_rate)

        print("Test Error Rate:{}".format(np.mean(miss_rate_list)))

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--show_graph', default=False, action='store_true')
    config = parser.parse_args()

    hebbLearner = HebbLearner(config)
    data, label = hebbLearner.create_dataset()

    # divide data into test and training
    total_num_data = data.shape[0]
    num_train_data = int(total_num_data*0.8)
    train_data, train_label = data[:num_train_data,:], label[:num_train_data]
    test_data, test_label = data[num_train_data:,:], label[num_train_data:]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hebbLearner.train(sess,train_data,train_label)
        hebbLearner.test(sess,test_data,test_label)
