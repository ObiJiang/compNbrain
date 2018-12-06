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
        self.batch_size = 100
        self.lr = 0.01
        self.alpha = 0.2
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
        return tf.tanh(val)
        #return tf.sign(tf.nn.relu(val))

    def model(self, bio_model = 'bdm'):
        sequences = tf.placeholder(tf.float32, [None, 30])
        labels = tf.placeholder(tf.int32, [None])
        real_labels = tf.expand_dims(labels,axis=1)
        #one_hot_labels = tf.one_hot(labels,2)

        """ forward pass """
        layer_sizes = [32,64,32]
        with tf.variable_scope('fw'):
            W_1 = self._weightVar([30,layer_sizes[0]],name='W1')
            b_1 = self.biasVar([layer_sizes[0]],name='b1')
            pre_y_1 = tf.matmul(sequences,W_1)
            y_1 = self.activation(pre_y_1-b_1)

            # second layer
            W_2 = self._weightVar([layer_sizes[0],layer_sizes[1]],name='W2')
            b_2 = self.biasVar([layer_sizes[1]],name='b2')
            pre_y_2 = tf.matmul(y_1,W_2)
            y_2= self.activation(pre_y_2-b_2)


            # third layer
            W_3 = self._weightVar([layer_sizes[1],layer_sizes[2]],name='W3')
            b_3 = self.biasVar([layer_sizes[2]],name='b3')
            pre_y_3 = tf.matmul(y_2,W_3)
            y_3= self.activation(pre_y_3-b_3)

            # third layer
            W_4 = self._weightVar([layer_sizes[2],1],name='W4')
            y_4 = tf.sign(tf.nn.relu(tf.matmul(y_3,W_4)))

            prediction = y_4

            miss_list = tf.not_equal(tf.cast(prediction,tf.float64),tf.cast(real_labels,tf.float64))
            correct_list = tf.equal(tf.cast(prediction,tf.float64),tf.cast(real_labels,tf.float64))
            dop_mask = tf.cast(correct_list,tf.float32)
            miss_rate = tf.reduce_sum(tf.cast(miss_list,tf.float32))/(self.batch_size)

        with tf.variable_scope('opt'):
            update_ops = []
            if bio_model == 'hebb':
                update_ops.extend([W_4.assign(W_4+self.lr*self.update_rule(y_3,tf.cast(2*real_labels-1,tf.float32)))])
                update_ops.extend([W_3.assign(W_3+self.lr*self.update_rule(y_2,y_3,dop_mask))])
                update_ops.extend([W_2.assign(W_2+self.lr*self.update_rule(y_1,y_2,dop_mask))])
                update_ops.extend([W_1.assign(W_1+self.lr*self.update_rule(sequences,y_1,dop_mask))])
                update_ops.extend([b_3.assign(self.alpha*b_3+(1-self.alpha)*tf.reduce_mean(pre_y_3,axis=0))])
                update_ops.extend([b_2.assign(self.alpha*b_2+(1-self.alpha)*tf.reduce_mean(pre_y_2,axis=0))])
                update_ops.extend([b_1.assign(self.alpha*b_1+(1-self.alpha)*tf.reduce_mean(pre_y_1,axis=0))])
            elif bio_model == 'ltd':
                theta = 1e-3
                update_ops.extend([W_4.assign(W_4+self.lr*self.update_rule(y_3,tf.cast(2*real_labels-1,tf.float32) - theta))])
                update_ops.extend([W_3.assign(W_3+self.lr*self.update_rule(y_2,y_3 - theta,dop_mask))])
                update_ops.extend([W_2.assign(W_2+self.lr*self.update_rule(y_1,y_2 - theta,dop_mask))])
                update_ops.extend([W_1.assign(W_1+self.lr*self.update_rule(sequences,y_1 - theta,dop_mask))])
                update_ops.extend([b_3.assign(self.alpha*b_3+(1-self.alpha)*tf.reduce_mean(pre_y_3,axis=0))])
                update_ops.extend([b_2.assign(self.alpha*b_2+(1-self.alpha)*tf.reduce_mean(pre_y_2,axis=0))])
                update_ops.extend([b_1.assign(self.alpha*b_1+(1-self.alpha)*tf.reduce_mean(pre_y_1,axis=0))])
            elif bio_model == 'bdm':
                theta = 1e-3
                update_ops.extend([W_4.assign(W_4+self.lr*self.update_rule(y_3,tf.cast(2*real_labels-1,tf.float32) * (tf.cast(2*real_labels-1,tf.float32) - theta)))])
                update_ops.extend([W_3.assign(W_3+self.lr*self.update_rule(y_2,y_3 * (y_3 - theta),dop_mask))])
                update_ops.extend([W_2.assign(W_2+self.lr*self.update_rule(y_1,y_2 * (y_2 - theta),dop_mask))])
                update_ops.extend([W_1.assign(W_1+self.lr*self.update_rule(sequences,y_1 * (y_1 - theta),dop_mask))])
                update_ops.extend([b_3.assign(self.alpha*b_3+(1-self.alpha)*tf.reduce_mean(pre_y_3,axis=0))])
                update_ops.extend([b_2.assign(self.alpha*b_2+(1-self.alpha)*tf.reduce_mean(pre_y_2,axis=0))])
                update_ops.extend([b_1.assign(self.alpha*b_1+(1-self.alpha)*tf.reduce_mean(pre_y_1,axis=0))])
            with tf.control_dependencies([y_4]):
                backwards_op = tf.tuple(update_ops)

        return AttrDict(locals())

    def update_rule(self,x,y,dop_mask=None):
        if dop_mask == None:
            return tf.matmul(tf.transpose(x),(y))
        else:
            return tf.matmul(tf.transpose(x),(y))
            #return tf.matmul(tf.transpose(x),dop_mask*(y))
            # temp = tf.einsum('bi,bj->bij', x, y)*(2*tf.expand_dims(dop_mask,axis=2)-1)
            # return tf.reduce_sum(temp,axis=0)

    def train(self,sess,data,labels):
        model = self.model
        for epoch_ind in range(100):
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

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hebbLearner.train(sess,X_train,y_train)
        hebbLearner.test(sess,X_test,y_test)
