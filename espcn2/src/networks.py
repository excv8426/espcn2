import tensorflow as tf
import numpy as np
import traceback
import logging

def _get_weights(config):
  weights={
    'w1': tf.Variable(tf.random_normal([5, 5, config.c_dim, 64], stddev=np.sqrt(2.0/25/3)), name='w1'),
    'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0/9/64)), name='w2'),
    'w3': tf.Variable(tf.random_normal([3, 3, 32, config.c_dim * config.scale * config.scale ], stddev=np.sqrt(2.0/9/32)), name='w3')
    }
  return weights

def _get_biases(config):
  biases={
    'b1': tf.Variable(tf.zeros([64], name='b1')),
    'b2': tf.Variable(tf.zeros([32], name='b2')),
    'b3': tf.Variable(tf.zeros([config.c_dim * config.scale * config.scale ], name='b3'))
    }
  return biases

def _phase_shift(config, I, r):
  # Helper function with main phase shift operation
  bsize, a, b, c = I.get_shape().as_list()
  X = tf.reshape(I, (config.batch_size, a, b, r, r))
  X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
  X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
  X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
  X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
  return tf.reshape(X, (config.batch_size, a*r, b*r, 1))


def _phase_shift_test(I ,r):
  bsize, a, b, c = I.get_shape().as_list()
  X = tf.reshape(I, (1, a, b, r, r))
  X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
  X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
  X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
  X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
  return tf.reshape(X, (1, a*r, b*r, 1))
          
def PS(config,X, r):
  # Main OP that you can arbitrarily use in you tensorflow code
  Xc = tf.split(X, 3, 3)
  if config.is_train:
    X = tf.concat([_phase_shift(config,x, r) for x in Xc], 3) # Do the concat RGB
  else:
    X = tf.concat([_phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
  return X


def espcn(config,normal_input):
  conv1 = tf.nn.relu(tf.nn.conv2d(normal_input, _get_weights(config)['w1'], strides=[1,1,1,1], padding='SAME') + _get_biases(config)['b1'])
  conv2 = tf.nn.relu(tf.nn.conv2d(conv1, _get_weights(config)['w2'], strides=[1,1,1,1], padding='SAME') + _get_biases(config)['b2'])
  conv3 = tf.nn.conv2d(conv2, _get_weights(config)['w3'], strides=[1,1,1,1], padding='SAME') + _get_biases(config)['b3'] # This layer don't need ReLU
  ps = PS(config,conv3, config.scale)
  ps=tf.nn.tanh(ps)
  return ps
