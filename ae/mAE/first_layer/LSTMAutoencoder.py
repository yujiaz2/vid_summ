import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell

import numpy as np

"""
Future : Modularization
"""

class LSTMAutoencoder(object):
  """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)

  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

  def __init__(self, hidden_num, inputs, 
    sparsity_level = 0.01, sparse_reg = 0.05, cell=None, optimizer=None, reverse=True, 
    decode_without_input=False):

    print 'LSTMae1'

    """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size 
              (batch_num x elem_num)
      cell : an rnn cell object (the default option 
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """

    self.batch_num = inputs[0].get_shape().as_list()[0]
    self.elem_num = inputs[0].get_shape().as_list()[1]

#    print inputs[0]
#    print inputs[0].get_shape()
#    print inputs[0].get_shape().as_list()[0]

    if cell is None:
      self._enc_cell = LSTMCell(hidden_num)
      self._dec_cell = LSTMCell(hidden_num)
    else :
      self._enc_cell = cell
      self._dec_cell = cell

    with tf.variable_scope('encoder'):
      self.z_codes, self.enc_state = tf.nn.rnn(
        self._enc_cell, inputs, dtype=tf.float32)
      # print self.z_codes

    with tf.variable_scope('decoder') as vs:
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32),
        name="dec_weight")
      dec_bias_ = tf.Variable(
        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
        name="dec_bias")

      # KL divergence
#      self.sparsity_level= np.repeat([0.05], hidden_num).astype(np.float32)
#      print enc_state.h
#      kl_div = self.kl_divergence(self.sparsity_level, enc_state.h)
#      print kl_div

      if decode_without_input:
        dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                      for _ in range(len(inputs))]
        dec_outputs, dec_state = tf.nn.rnn(
          self._dec_cell, dec_inputs, 
          initial_state=self.enc_state, dtype=tf.float32)
#	print dec_state
	"""the shape of each tensor
          dec_output_ : (step_num x hidden_num)
          dec_weight_ : (hidden_num x elem_num)
          dec_bias_ : (elem_num)
          output_ : (step_num x elem_num)
          input_ : (step_num x elem_num)
        """
        if reverse:
          dec_outputs = dec_outputs[::-1]
        dec_output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])
        dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num,1,1])
        self.output_ = tf.batch_matmul(dec_output_, dec_weight_) + dec_bias_

      else : 
#	enc_state.h = kl_div
        # print np.shape(self.enc_state)
        enc_state = self.enc_state[:,2048:4096]
        dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_outputs = []
        for step in range(len(inputs)):
          if step>0: vs.reuse_variables()
          dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
          dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
          dec_outputs.append(dec_input_)
          # print np.shape(dec_input_)
        if reverse:
          dec_outputs = dec_outputs[::-1]
        self.output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])

#    print self.z_codes
#    print type(enc_state.h)
    self.input_ = tf.transpose(tf.pack(inputs), [1,0,2])
#    self.sparsity_level= np.repeat([0.05], hidden_num).astype(np.float32)
    self.sparsity_level=np.tile([0.05], (self.batch_num, hidden_num)).astype(np.float32)
    kl_div =self.kl_divergence(self.sparsity_level, self.enc_state[:,2048:4096])
#    self.sparse_reg = sparse_reg
    self.sparse_reg = 0.001
#    print np.type(enc_state.h)
#    sum_ = np.sum(kl_div)
#    print sum
    self.p_hat = self.enc_state[:,2048:4096]
    self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_)) + self.sparse_reg*tf.reduce_sum(kl_div)
#    self.loss = kl_div
#    print type(kl_div)
#    print np.shape(self.loss)
#    print 'test'

    if optimizer is None :
      self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
    else :
      self.train = optimizer.minimize(self.loss)

  def kl_divergence(self, p, p_hat):
#        print np.shape(p)
#	p_hat = (p_hat+1)/2
	return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)
