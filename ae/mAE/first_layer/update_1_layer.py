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

batch_num = 1
hidden_num = 2048
step_num = 3
elem_num = 8192

begin = False

p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

model_path = '/home/dingwen/meanMask/obj_cont/mAE/first_layer/obj_cont_1_layer.ckpt'
saver = tf.train.Saver()

feat_num_all = np.loadtxt('/home/dingwen/meanMask/obj_cont/online_feat/feat_9parts.txt')
index = 0


for no in range(1,30+1):
    feat_path = '/home/dingwen/meanMask/obj_cont/online_feat/meanFeat_online'+'%02d' %no +'.mat'
    count = len(open(feat_path,'rU').readlines())
    clip_num = count/step_num

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        input_ = sio.loadmat(feat_path)
        input_ = input_['feat']
        input_ = input_[0:clip_num*step_num,:]
        input_ = np.reshape(input_,(-1,step_num,elem_num))

        for inindex in range (0,9):

            if(index%9 == 0):
                feat_input = input_[0:int(feat_num_all[index]),:]
            else:
                feat_input = input_[int(feat_num_all[index-1]):int(feat_num_all[index]),:]

            train_cnt = np.size(feat_input,0)

            print 'feat_num: ' + str(train_cnt)

            summary_cur = []

            for train_iter in range(0,train_cnt):
                train_input = feat_input[train_iter:train_iter+batch_num]
                train_iter = train_iter+batch_num

                concate = False
                loss0 = 0
                for out_tensor in ae.z_codes:
                    loss_, result = sess.run([ae.loss, out_tensor], {p_input:train_input})

                    if concate == False:
                        concate = True
                        loss0 = loss_
                    else:
                        loss0 = loss0+loss_
                loss_final = loss0/step_num

                if index%9 == 0:
                    min_ = 0.0135
                    max_ = 0.1489

                loss_norm = (loss_final-min_)/(max_-min_)

                if ((loss_final-min_)/(max_-min_)>=0):
                    if begin==False:
                        begin = True
                        summary_cur = train_input
                    else:
                        summary_cur = np.concatenate((summary_cur,train_input))

                filename2 = 'loss1.txt'
                f2 = open(filename2,'a')
                loss_Final = [loss_norm]
                np.savetxt(f2,loss_Final)
                f2.close()

            summ_cnt = np.size(summary_cur,0)


            for train_times in range(0,2):
                for update_iter in range(0,train_cnt):
                    update_input = feat_input[update_iter:update_iter+batch_num]
                    update_iter = update_iter+batch_num
                    loss, __,result = sess.run([ae.loss, ae.train,out_tensor], {p_input:update_input})
                    print train_times
                    print loss

            for summary_iter in range(0,train_cnt):

                summ_input = feat_input[summary_iter:summary_iter+batch_num]
                summary_iter = summary_iter+batch_num

                concate2 = False
                loss1 = 0
                for out_tensor in ae.z_codes:
                    loss, result = sess.run([ae.loss, out_tensor], {p_input:summ_input})
                    result = np.reshape(result,(batch_num,1,hidden_num))
                    if concate2 == False:
                        res = result
                        concate2 = True
                        loss1 = loss
                    else:
                        loss1 = loss1+loss
                        res = np.concatenate((res,result),axis=1)
                loss_final1 = loss1/step_num

                res_ = np.reshape(res,(batch_num*step_num,-1))
                save_file_ = '/home/dingwen/meanMask/obj_cont/online_feat/feat_obj_cont_online_2048_'+'%02d' %no +'.txt'
                f_ = open(save_file_,'a')
                np.savetxt(f_, res_)
                print save_file_

                if loss_final1 < min_:
                    min_ = loss_final1
                if loss_final1 > max_:
                    max_ = loss_final1

            begin = False
            index = index+1
        print '~~~~~~~~~~~~~~~~~~~~~~~'
            
