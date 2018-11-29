import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class HebbLearner():
    def __init__(self,config):
        self.num_sequence = 1000
        self.config = config
        self.batch_size = 1000
        self.lr = 0.01
        self.model = self.model()

    def _weightVar(self,shape, mean=0.0, stddev=0.01, name='weights'):
        init_w = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        weights = tf.get_variable(name=name,initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=stddev))

        return weights

    def model(self):
        sequences = tf.placeholder(tf.float32, [None, 28,28])
        labels = tf.placeholder(tf.int32, [None])

        sequences_flat = tf.reshape(sequences,[-1,28*28])
        one_hot_labels = tf.one_hot(labels,10)

        """ forward pass """
        # simple archecture 2->32->32->2
        # activation: relu
        # first layer
        with tf.variable_scope('fw'):
            W_1 = self._weightVar([28*28,64],name='W1')
            y_1 = tf.nn.tanh(tf.matmul(sequences_flat,W_1))

            # second layer
            W_2 = self._weightVar([64,64],name='W2')
            y_2= tf.nn.tanh(tf.matmul(y_1,W_2))

            # third layer
            W_3 = self._weightVar([64,64],name='W3')
            y_3= tf.nn.tanh(tf.matmul(y_2,W_3))

            # third layer
            W_4 = self._weightVar([64,10],name='W4')
            y_4 = tf.matmul(y_3,W_4)

            prediction = tf.argmax(y_4,axis=1)

            miss_list = tf.not_equal(tf.cast(prediction,tf.float64),tf.cast(labels,tf.float64))
            miss_rate = tf.reduce_sum(tf.cast(miss_list,tf.float32))/(self.batch_size)

        update_ops = []
        with tf.variable_scope('bw'):
            #[W_3.assign(W_3+self.lr*tf.matmul(tf.transpose(y_2),tf.cast(tf.expand_dims(labels,1),tf.float32)))])
            update_ops.extend([W_4.assign(W_4+self.lr*tf.matmul(tf.transpose(y_3),tf.cast(2*one_hot_labels-1,tf.float32)))])
            update_ops.extend([W_3.assign(W_3+self.lr*tf.matmul(tf.transpose(y_2),y_3))])
            update_ops.extend([W_2.assign(W_2+self.lr*tf.matmul(tf.transpose(y_1),y_2))])
            update_ops.extend([W_1.assign(W_1+self.lr*tf.matmul(tf.transpose(sequences_flat),y_1))])
            with tf.control_dependencies([y_4]):
                backwards_op = tf.tuple(update_ops)

        return AttrDict(locals())

    def train(self,sess,data,labels):
        model = self.model
        for epoch_ind in range(50):
            miss_rate_list = []
            for bi in range(int(data.shape[0]/self.batch_size)):
                batch_data = data[bi*self.batch_size:(bi+1)*self.batch_size,:]
                batch_labels = labels[bi*self.batch_size:(bi+1)*self.batch_size]
                W,miss_rate = sess.run([model.backwards_op,model.miss_rate],feed_dict={model.sequences:batch_data,model.labels:batch_labels})
                miss_rate_list.append(miss_rate)

            print("Tranning Error at Epochs {}:{}".format(epoch_ind,np.mean(miss_rate_list)))

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

    # divide data into test and training
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data, test_data = ((train_data -128)/ 255.0).astype(np.int32), ((test_data -128)/ 255.0).astype(np.int32)

    # train_label_unpack = np.unpackbits(np.expand_dims(train_label,axis=1), axis=1)[:,-4:]
    # test_label_unpack = np.unpackbits(np.expand_dims(test_label,axis=1), axis=1)[:,-4:]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hebbLearner.train(sess,train_data,train_label)
        hebbLearner.test(sess,test_data,test_label)
