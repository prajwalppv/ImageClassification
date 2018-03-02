import os
import sys

import numpy as np
import tensorflow as tf

from src.utils import count_model_params

from src.data_utils import create_batches

class Hparams:
  # number of output classes. Must be 10 for CIFAR-10
  num_classes = 10

  # size of each train mini-batch
  batch_size =  1000    #1000 gave 71%

  # size of each eval mini-batch
  eval_batch_size = 100

  learning_rate = 0.1

  # l2 regularization rate
  l2_reg = 1e-4

  # number of training steps
  train_steps = 20000


def conv_net(images, labels, *args, **kwargs):
  """A conv net.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """
  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  def _get_logits(x,flag=True):
    H, W, C = (x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value)
    weights = []
    x = tf.reshape(x, [-1, H, W, C])

    # COnv layer 1
    with tf.variable_scope("conv_1", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", [3, 3, C, 32])
      # weights.append(w)
    x = tf.nn.conv2d(x, w, [1, 2, 2, 1], padding="SAME")
    x = tf.nn.relu(x)

    # Max pooling layer 1
    # x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    # Conv layer train 2
    H, W, C = (x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value)
    with tf.variable_scope("conv_2", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", [3, 3, C, 64])
    x = tf.nn.conv2d(x, w, [1, 2, 2, 1], padding="SAME")
    x = tf.nn.relu(x)

    # Max pool layer 2
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME")

    # COnv layer 1
    H, W, C = (x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value)
    with tf.variable_scope("conv_3", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", [3, 3, C, 64])
      # weights.append(w)
    x = tf.nn.conv2d(x, w, [1, 2, 2, 1], padding="SAME")
    x = tf.nn.relu(x)

    # Max pooling layer 1
    # x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    # Conv layer train 2
    H, W, C = (x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value)
    with tf.variable_scope("conv_4", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", [3, 3, C, 64])
    x = tf.nn.conv2d(x, w, [1, 2, 2, 1], padding="SAME")
    # x = tf.nn.dropout(x,keep_prob=0.8)
    x = tf.nn.relu(x)

    # Max pool layer 2
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME")


    H, W, C = (x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value)
    x = tf.reshape(x, [-1, H * W * C])

    with tf.variable_scope("dense2", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", shape=[x.get_shape()[-1].value, 512])

    x = tf.matmul(x, w)

    with tf.variable_scope("dense3", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", shape=[x.get_shape()[-1].value, 256])

    x = tf.matmul(x, w)

    with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", shape=[x.get_shape()[-1].value, hparams.num_classes])

    logits = tf.matmul(x, w)
    return logits


  train_logits = _get_logits(x_train,True)
  valid_logits = _get_logits(x_valid)
  test_logits  = _get_logits(x_test)

  global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=y_train)
  train_loss = tf.reduce_mean(log_probs) #+ tf.losses.get_regularization_loss()
  exp_decay = tf.train.exponential_decay(0.1,global_step,15000,0.05)
  optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(test_logits, y_test)
  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }

  return ops


def feed_forward_net(images, labels, *args, **kwargs):
  """A feed_forward_net.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """
  # dims = kwargs[0]

  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  # create model parameters
  def _get_logits(x,dims=[512,512,256],flag=False,weights=None):
    H, W, C = (x.get_shape()[1].value, x.get_shape()[2].value,x.get_shape()[3].value)
    if weights==None:
      weights = []
    x = tf.reshape(x,[-1 ,H * W * C ]) # flatten
    for layer_id,next_dim in enumerate(dims):
      curr_dim = x.get_shape()[-1].value # get_shape () returns a <list >
      # print(layer_id)
      if flag:
        with tf.variable_scope("layer_{}".format(layer_id)):
          w = tf.get_variable ("w", [curr_dim,next_dim]) # w’s name : " layer_2 /w"
        weights.append(w)
      else:
          w = weights[layer_id]
      x = tf.matmul(x , w)
      x = tf.nn.relu(x)

    curr_dim = x.get_shape()[-1].value # get_shape () returns a <list > 
    if flag: 
      with tf.variable_scope("logits"):
          w = tf.get_variable("w",[curr_dim,hparams.num_classes]) # w’s name : " logits /w"
          weights.append(w)
    else:
        w = weights[-1]

    logits = tf.matmul(x,w)
    return logits,weights
  
  train_logits, weights = _get_logits(x_train,flag=True)
  valid_logits,_ = _get_logits(x_valid,weights=weights)
  test_logits,_ = _get_logits(x_test,weights=weights)

  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,name="global_step")
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=y_train)
  train_loss = tf.reduce_mean(log_probs)
  optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(test_logits, y_test)
  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }

  return ops


def softmax_classifier(images, labels, name="softmax_classifier"):
  """A softmax classifier.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """

  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  H, W, C = (x_train.get_shape()[1].value,
             x_train.get_shape()[2].value,
             x_train.get_shape()[3].value)

  # create model parameters
  with tf.variable_scope(name):
    w_soft = tf.get_variable("w", [H * W * C, hparams.num_classes])

  # compute train, valid, and test logits
  def _get_logits(x):
    x = tf.reshape(x, [-1, H * W * C])
    logits = tf.matmul(x, w_soft)
    return logits

  train_logits = _get_logits(x_train)
  valid_logits = _get_logits(x_valid)
  test_logits = _get_logits(x_test)

  # create train_op and global_step
  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                            name="global_step")
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=train_logits, labels=y_train)
  train_loss = tf.reduce_mean(log_probs)
  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(test_logits, y_test)

  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }
  return ops

