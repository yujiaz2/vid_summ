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
hidden_num = 1024
step_num = 1
elem_num = 4096

begin = False

p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

model_path = '/home/yjzhang/exp_feat/context_feat/first_layer/con_1_layer.ckpt'
saver = tf.train.Saver()

feat_num_all = np.loadtxt('/home/yjzhang/ae_concatenate/third_layer/feat_9parts.txt')
index = 0


for no in range(1,30+1):
    feat_path = '/home/yjzhang/feat/feat_online'+'%02d' %no +'.txt'
    count = len(open(feat_path,'rU').readlines())
    clip_num = count/step_num

    # print 'clip_num: ' + str(clip_num)
# Constants
    

    with tf.Session() as sess:
        saver.restore(sess, model_path)
    #path = '/home/yjzhang/ae_offline/second_layer/'
    #path_ = path + 'feat_onlineTest_512.txt'

    #path0 = '/home/yjzhang/ae_offline/third_layer/'


    #if os.path.exists(path):

        input_ = np.loadtxt(feat_path,dtype = 'float')
        input_ = input_[:,4096:]
        input_ = input_[0:clip_num*step_num,:]
        input_ = np.reshape(input_,(-1,step_num,elem_num))
        # print np.shape(input)

        # print 'input.shape: '
        # print np.shape(input)

        for inindex in range (0,9):

            # print 'inindex: ' + str(inindex)
            # print index
            # print int(feat_num_all[index])

            if(index%9 == 0):
                feat_input = input_[0:int(feat_num_all[index]),:]
            else:
                feat_input = input_[int(feat_num_all[index-1]):int(feat_num_all[index]),:]
#            print np.shape(feat_input)

            # print feat_num_all[index-1]
            # print feat_num_all[index]

            train_cnt = np.size(feat_input,0)
            # print int(feat_num_all[index])   22
            # print np.shape(feat_input)   5,3,512

            print 'feat_num: ' + str(train_cnt)

            summary_cur = []

            for train_iter in range(0,train_cnt):
                train_input = feat_input[train_iter:train_iter+batch_num]
                train_iter = train_iter+batch_num
                # print np.shape(train_input)

                concate = False
                loss0 = 0
                for out_tensor in ae.z_codes:
                    loss_, result = sess.run([ae.loss, out_tensor], {p_input:train_input})
                # loss, __,result = sess.run([ae.loss, ae.train,out_tensor], {p_input:train_input})
                    if concate == False:
                        concate = True
                        loss0 = loss_
                    else:
                        loss0 = loss0+loss_
                loss_final = loss0/step_num

                # print 'loss_final: ' + str(loss_final)
            # print loss_final

                if index%9 == 0:
                    min_ = 0.0018
                    max_ = 0.1042

                loss_norm = (loss_final-min_)/(max_-min_)

                if ((loss_final-min_)/(max_-min_)>=0):
                    if begin==False:
                        begin = True
                        summary_cur = train_input
                    else:
                        summary_cur = np.concatenate((summary_cur,train_input))

                    # print 'summary_cur.shape: '
                    # print np.shape(summary_cur)

                    filename1 = 'summary_res_1.txt'
                    f1 = open(filename1,'a')
                    a = [1]
                    np.savetxt(f1,a,fmt="%d")
                    f1.close()
                    
                else:
                    filename3 = 'summary_res_1.txt'
                    f3 = open(filename3,'a')
                    b = [0]
                    np.savetxt(f3,b,fmt="%d")
                    f3.close()
                filename2 = 'loss_res_1.txt'
                f2 = open(filename2,'a')
                loss_Final = [loss_norm]
                np.savetxt(f2,loss_Final)
                f2.close()

            summ_cnt = np.size(summary_cur,0)

            print 'summ_cnt: ' + str(summ_cnt)

            for train_times in (0,10):
                for update_iter in range(0,train_cnt):
                    update_input = feat_input[update_iter:update_iter+batch_num]
                    update_iter = update_iter+batch_num
                    loss, __,result = sess.run([ae.loss, ae.train,out_tensor], {p_input:update_input})

            for summary_iter in range(0,train_cnt):
                    # for cnt in range(0,summ_cnt):
                    #     summ_tmp = summary_cur[cnt:cnt+batch_num]
                    #     cnt = cnt+batch_num
                    #     loss, __,result = sess.run([ae.loss, ae.train,out_tensor], {p_input:summ_tmp})

                summ_input = feat_input[summary_iter:summary_iter+batch_num]
                summary_iter = summary_iter+batch_num

                   # min = 1.0
                   # max = 0.0

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
                save_file_ = '/home/yjzhang/feat/feat_con_online_1024_'+'%02d' %no +'.txt'
                f_ = open(save_file_,'a')
                np.savetxt(f_, res_)
                print save_file_
                # print 'loss_final1'
                # print 'loss'
                #     if summary_iter==1:
                #         min_ = loss_final1
                #         max_ = loss_final1
                #     else:
                if loss_final1 < min_:
                    min_ = loss_final1
                if loss_final1 > max_:
                    max_ = loss_final1
                # print min
                # print max
            begin = False
            index = index+1
        print '~~~~~~~~~~~~~~~~~~~~~~~'
            
