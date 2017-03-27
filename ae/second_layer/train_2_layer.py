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
batch_num = 200
hidden_num = 512
step_num = 3
elem_num = 1024
iteration = 1360#epoch = 884*200/6800 =26 
input_iter = 0
clip_num = 6800
iter_per_epoch = clip_num/batch_num
iter_15epoch = iter_per_epoch*15

# placeholder list
p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

model_path = sys.path[0] + '/con_2_layer.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    path = '/home/yjzhang/exp_feat/context_feat/first_layer/'
    path_ = path + 'feat_con_1024.txt'


    if os.path.exists(path_):
        # clip_num = 20400/3

        input_ = np.loadtxt(path_,dtype = 'float')
        input_ = input_[0:clip_num*step_num,:]
        input_ = np.reshape(input_,(-1,step_num,elem_num))

        train_input = input_[0:clip_num/batch_num*batch_num,:,:]
        update_iter = clip_num/batch_num
  	
        for i in range(iteration):
            if ((i % update_iter == 0) and (i > 0)):
                input_iter = 0

            batch_input = train_input[input_iter:input_iter+batch_num]
            input_iter = input_iter+batch_num

            loss_val, _ = sess.run([ae.loss, ae.train], {p_input:batch_input})
           
        save_path = saver.save(sess, model_path)
        print "[+] Model saved in file: %s" % save_path

    # test_cnt = clip_num/batch_num
    # for test_iter in range(test_cnt):
    #     test_input = train_input[test_iter:test_iter+batch_num]
    #     test_iter = test_iter+batch_num

    #     concate = False
    #     for out_tensor in ae.z_codes:
    #         loss, result = sess.run([ae.loss, out_tensor], {p_input:test_input})
    #         result = np.reshape(result,(batch_num,1,hidden_num))
    #         if concate == False:
    #             res = result
    #             concate = True
    #         else:
    #             res = np.concatenate((res,result),axis=1)
    #     print np.shape(res)


    # #     # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # #     # print np.shape(result)
    # #         # print np.shape(result)

    #     res_ = np.reshape(res,(batch_num*step_num,-1))
    #     print np.shape(res_)
    #     path__ = '/home/yjzhang/exp_feat/obj_feat/second_layer/'
    #     f = open(path__+'feat_obj_512.txt','a')
    #     np.savetxt(f, res_)
    # # path2 = "/home/yjzhang/ae_offline/first_layer/"
    # # path2_ = path2 + 'feat_1024.txt'
    # # test = np.loadtxt(path2_)
    # # print np.shape(test)
