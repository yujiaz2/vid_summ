# Basic libraries
import numpy as np
import tensorflow as tf
import os
import math
import random
import sys
import scipy.io as sio

tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import *

# Constants
batch_num = 20
hidden_num = 2048
step_num = 3
elem_num = 8192
iteration = 792 # 25 epoch
input_iter = 0

# placeholder list
p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

model_path = sys.path[0] + '/obj_cont_1_layer.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    path = '/home/dingwen/meanMask/obj_cont/find_iter/first_layer/'
    path_ = path + 'feat_all_mean.mat'

    if os.path.exists(path_):
        clip_num = 6600

        input_ = sio.loadmat(path_)
        input_ = input_['feat']
        input_ = input_[0:clip_num*step_num,:]
        input_ = np.reshape(input_,(-1,step_num,elem_num))

        train_input = input_[0:clip_num/batch_num*batch_num,:,:]
        update_iter = clip_num/batch_num
  	
        for i in range(0,iteration):
            if ((i % update_iter == 0) and (i > 0)):
                input_iter = 0

            batch_input = train_input[input_iter:input_iter+batch_num]
            input_iter = input_iter+batch_num

            loss_val, _ = sess.run([ae.loss, ae.train], {p_input:batch_input})
            
        save_path = saver.save(sess, model_path)
        print "[+] Model saved in file: %s" % save_path
