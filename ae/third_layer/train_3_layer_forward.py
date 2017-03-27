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
from LSTMAutoencoder3 import *

# Constants
batch_num = 1
hidden_num = 256
step_num = 3
elem_num = 512


# placeholder list
p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

model_path = '/home/yjzhang/exp_feat/context_feat/third_layer/con_3_layer.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    path = '/home/yjzhang/exp_feat/context_feat/second_layer/'
    path_ = path + 'feat_con_512.txt'
    path0 = '/home/yjzhang/exp_feat/context_feat/third_layer/'


    if os.path.exists(path):
        # f = open(path_)
        # rows_len = len(open(path_,'rU').readlines())
        clip_num = 6800
        # f.close()

        input = np.loadtxt(path_,dtype = 'float')
        input = input[0:clip_num*step_num,:]
        input = np.reshape(input,(-1,step_num,elem_num))

        train_cnt = clip_num/batch_num
        for train_iter in range(train_cnt):
            train_input = input[train_iter:train_iter+batch_num]
            train_iter = train_iter+batch_num

            concate = False
            for out_tensor in ae.z_codes:
                loss, result = sess.run([ae.loss, out_tensor], {p_input:train_input})
                result = np.reshape(result,(batch_num,1,hidden_num))
                if concate == False:
                    res = result
                    concate = True
                    loss0 = loss
                else:
                    res = np.concatenate((res,result),axis=1)
                    loss0 = loss0+loss
            loss_final = loss0/step_num
            print loss_final
            # res_ = np.reshape(res,(batch_num*step_num,-1))
            # print np.shape(res_)
            # f = open(path0+'feat_rand_256.txt','a')
            # np.savetxt(f, res_)
