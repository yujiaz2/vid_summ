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
batch_num = 200 #1epoch = 6800/100 = 68
hidden_num = 256
step_num = 3
elem_num = 512
iteration = 3300 # epoch=544*68/6800=8
input_iter = 0

# placeholder list
p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    path = '/home/dingwen/meanMask/obj_cont/mAE/second_layer/'
    path_ = path + 'feat_obj_cont_512.txt'
    path__ = path + 'feat_validation_512.txt'


    if os.path.exists(path_):
        clip_num = 6600

        input_ = np.loadtxt(path_,dtype = 'float')
        input_ = input_[0:clip_num*step_num,:]
        input_ = np.reshape(input_,(-1,step_num,elem_num))

        val_input = np.loadtxt(path__,dtype = 'float')
        val_input = val_input[0:200*step_num,:]
        val_input = np.reshape(val_input,(-1,step_num,elem_num))

        train_input = input_[0:clip_num/batch_num*batch_num,:,:]
        update_iter = clip_num/batch_num
  	
        for i in range(iteration):
            if ((i % update_iter == 0) and (i > 0)):
                input_iter = 0

                loss1 = sess.run([ae.loss], {p_input:val_input})
                print 'iter:' + str(i)+'loss: '+str(loss1)

            batch_input = train_input[input_iter:input_iter+batch_num]
            input_iter = input_iter+batch_num

            loss_val, _ = sess.run([ae.loss, ae.train], {p_input:batch_input})
