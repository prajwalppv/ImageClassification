from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _pickle as pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import count_model_params
from src.utils import print_user_flags

from src.data_utils import read_data

from src.models import Hparams
from src.models import softmax_classifier,feed_forward_net,conv_net

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", None, "Path to CIFAR-10 data")
DEFINE_string("output_dir", "output", "Path to log folder")

DEFINE_string("model_name", "",
              "Name of the method. [softmax|feed_forward|conv]")

DEFINE_integer("log_every", 10, "How many steps to log")

def get_ops(images, labels):
  """Builds the model."""

  print("-" * 80)
  print("Creating a '{0}' model".format(FLAGS.model_name))
  if FLAGS.model_name == "softmax":
    ops = softmax_classifier(images, labels)
  elif FLAGS.model_name == "feed_forward":
    ops = feed_forward_net(images, labels)
  elif FLAGS.model_name == "conv_net":
    ops = conv_net(images, labels)
  else:
    raise ValueError("Unknown model name '{0}'".format(FLAGS.model_name))

  assert "global_step" in ops
  assert "train_op" in ops
  assert "train_loss" in ops
  assert "valid_acc" in ops
  assert "test_acc" in ops

  return ops


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {0} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {0} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print_user_flags()

  hparams = Hparams()
  images, labels = read_data(FLAGS.data_path)

  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels)

    # count model variables
    tf_variables = tf.trainable_variables()
    saver = tf.train.Saver()
    num_params = count_model_params(tf_variables)

    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            # pylint: disable=W0212
            session = session._sess
        return session

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=FLAGS.output_dir) as sess:

        # training loop
        print("-" * 80)
        print("Starting training")
        for step in range(1, hparams.train_steps + 1):
          sess.run(ops["train_op"])
          if step % FLAGS.log_every == 0:
            global_step, train_loss, valid_acc = sess.run([
              ops["global_step"],
              ops["train_loss"],
              ops["valid_acc"],
            ])
            log_string = ""
            log_string += "step={0:<6d}".format(step)
            log_string += " loss={0:<5.2f}".format(train_loss)
            log_string += " val_acc={0:<3d}/{1:<3d}".format(
              valid_acc, hparams.eval_batch_size)
            print(log_string)
            sys.stdout.flush()
          if step%10000 ==0 :
            path = "./models/{0}/ShallowModel_{1}.cpkt".format(step,step)
            saver.save(get_session(sess), path)

        # old_sess = sess
        # final test
        print("-" * 80)
        print("Training done. Eval on TEST set")
        num_corrects = 0
        for _ in range(10000 // hparams.eval_batch_size):
          num_corrects += sess.run(ops["test_acc"])
        print("test_accuracy: {0:>5d}/10000".format(num_corrects))
        # saver.save(get_session(sess), "./models/ShallowModel.cpkt")

if __name__ == "__main__":
  tf.app.run()
