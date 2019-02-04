# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 00:55:16 2018

@author: ngoro
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf

def customDNN(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        net = tf.layers.dropout(net, rate=0.1)
        
  
  # end the network with a denser layer of size 2.
  logits = tf.layers.dense(net,params['n_classes'], activation=None)

  # PREDICTIONS
  pred_class = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': pred_class,
        'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # LOSS
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # TRAINING
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # EVAL.
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=pred_class)
  }
  return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  
  
  
 
  
  
  
  
  
  
  
  
  
  
  