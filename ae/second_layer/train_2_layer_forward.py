# Basic libraries
import numpy as np
import tensorflow as tf
import os
import math
import random
import sys
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder2 import *

# Constants
batch_num = 1
hidden_num = 512
step_num = 3
elem_num = 1024


# placeholder list
p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

model_path = sys.path[0] + '/con_2_layer.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    path = '/home/yjzhang/exp_feat/context_feat/first_layer/'
    path_ = path + 'feat_con_1024.txt'
    path0 = '/home/yjzhang/exp_feat/context_feat/second_layer/'


    if os.path.exists(path):
        # f = open(path_)
        # rows_len = len(open(path_,'rU').readlines())
        clip_num = 6800
        # f.close()

        input_ = np.loadtxt(path_,dtype = 'float')
        input_ = input_[0:clip_num*step_num,:]
        input_ = np.reshape(input_,(-1,step_num,elem_num))
        # print np.shape(input_)

        train_cnt = clip_num/batch_num
        for train_iter in range(train_cnt):
            train_input = input_[train_iter:train_iter+batch_num]
            train_iter = train_iter+batch_num

            concate = False
            for out_tensor in ae.z_codes:
                loss, result = sess.run([ae.loss, out_tensor], {p_input:train_input})
                # print result
                # print np.shape(result)
                result = np.reshape(result,(batch_num,1,hidden_num))

                if concate == False:
                    res = result
                    concate = True
                    loss0 = loss
                else:
                    res = np.concatenate((res,result),axis=1)
                    loss0 = loss0 + loss
            loss_final = loss0/step_num
            print loss_final
            res_ = np.reshape(res,(batch_num*step_num,-1))
            # print np.shape(res_)
            f = open(path0+'feat_con_512.txt','a')
            np.savetxt(f, res_)