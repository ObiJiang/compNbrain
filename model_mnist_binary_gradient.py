import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
""" Import PCA-related stuff from sklearn """
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class HebbLearner():
    def __init__(self,config):
        self.num_sequence = 1000
        self.config = config
        self.batch_size = 1000
        self.lr = 0.01
        self.keep_prob = 0.5
        self.model = self.model()

    def _weightVar(self,shape, mean=0.0, stddev=0.01, name='weights'):
        init_w = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        weights = tf.get_variable(name=name,initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=stddev))

        return weights

    def biasVar(self, shape, value=0.05, name='bias'):
        init_b = tf.constant(value=value, shape=shape)
        b = tf.get_variable(name=name,initializer=tf.constant(value=value, shape=shape))
        tf.summary.histogram('b', b)
        return b

    def activation(self,val):
        return tf.nn.relu(tf.sign(val))

    def model(self):
        sequences = tf.placeholder(tf.float32, [None, 236])
        labels = tf.placeholder(tf.int32, [None])

        one_hot_labels = tf.one_hot(labels,10)

        """ forward pass """
        layer_sizes = [128,256,128]
        with tf.variable_scope('fw'):
            W_1 = self._weightVar([236,layer_sizes[0]],name='W1')
            b_1 = self.biasVar([layer_sizes[0]],name='b1')
            y_1 = self.activation(tf.matmul(sequences,W_1)+b_1)

            # second layer
            W_2 = self._weightVar([layer_sizes[0],layer_sizes[1]],name='W2')
            b_2 = self.biasVar([layer_sizes[1]],name='b2')
            y_2= self.activation(tf.matmul(y_1,W_2)+b_2)

            # third layer
            W_3 = self._weightVar([layer_sizes[1],layer_sizes[2]],name='W3')
            b_3 = self.biasVar([layer_sizes[2]],name='b3')
            y_3= self.activation(tf.matmul(y_2,W_3)+b_3)

            # third layer
            W_4 = self._weightVar([layer_sizes[2],10],name='W4')
            b_4 = self.biasVar([10],name='b4')
            y_4 = tf.matmul(y_3,W_4)+b_4

            prediction = tf.argmax(y_4,axis=1)

            miss_list = tf.not_equal(tf.cast(prediction,tf.float64),tf.cast(labels,tf.float64))
            miss_rate = tf.reduce_sum(tf.cast(miss_list,tf.float32))/(self.batch_size)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=y_4)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            # Compute the gradients for a list of variables.
            grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())

            # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
            # need to the 'gradient' part, for example cap them, etc.
            binarized_grads_and_vars = [(tf.nn.relu(tf.sign(gv[0])), gv[1]) for gv in grads_and_vars]

            # Ask the optimizer to apply the capped gradients.
            opt = optimizer.apply_gradients(binarized_grads_and_vars)
            #
            # opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

        return AttrDict(locals())

    def train(self,sess,data,labels):
        model = self.model
        for epoch_ind in range(50):
            miss_rate_list = []
            for bi in range(int(data.shape[0]/self.batch_size)):
                batch_data = data[bi*self.batch_size:(bi+1)*self.batch_size,:]
                batch_labels = labels[bi*self.batch_size:(bi+1)*self.batch_size]
                W,miss_rate,y4 = sess.run([model.opt,model.miss_rate,model.y_4],feed_dict={model.sequences:batch_data,model.labels:batch_labels})
                miss_rate_list.append(miss_rate)
                #print(y4)

            print("Tranning Error at Epochs {}:{}".format(epoch_ind,np.mean(miss_rate_list)))

    def test(self,sess,data,labels):
        self.keep_prob = 1.0
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
    train_data, test_data = (train_data)/ 255.0, (test_data)/ 255.0

    train_data = np.reshape(train_data,[-1,28*28])
    test_data = np.reshape(test_data,[-1,28*28])

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    pca = PCA(.9)
    pca.fit(train_data)
    pca_train_data = pca.transform(train_data)
    pca_test_data = pca.transform(test_data)

    print(pca_train_data.shape)

    # train_label_unpack = np.unpackbits(np.expand_dims(train_label,axis=1), axis=1)[:,-4:]
    # test_label_unpack = np.unpackbits(np.expand_dims(test_label,axis=1), axis=1)[:,-4:]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hebbLearner.train(sess,pca_train_data,train_label)
        hebbLearner.test(sess,pca_test_data,test_label)
