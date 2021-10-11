# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple FedAvg to train EMNIST.

This is intended to be a minimal stand-alone experiment script built on top of
core TFF.
"""

import collections
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_federated as tff

import simple_fedavg_tf
import simple_fedavg_tff

from create_datasets import create_federated_dataset, preprocess, preprocess2, preprocess_repeat
from resnet_models2 import create_res_net

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


# Training hyperparameters
flags.DEFINE_integer('num_clients', 5, 'Number of clients.')
flags.DEFINE_integer('total_num_clients', 5, 'Number of total clients for dataset creation.')
flags.DEFINE_integer('total_rounds', 5, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 128, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 128, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.01, 'Client learning rate.')

FLAGS = flags.FLAGS

tf.random.set_seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)


def create_original_fedavg_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  input_shape = [32, 32, 3]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer=initializer)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(10 if only_digits else 62, kernel_initializer=initializer),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


# def client_optimizer_fn():
#   return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate, momentum=0.9)


def client_optimizer_fn(lr):
  return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

# momentum ve momentumsuz 0.01 lr için dene

@tf.function
def lr_schedule(round_num):
  if round_num < 10:
    lr = 0.1
  elif round_num < 50:
    lr = 0.01
  else:
    lr = 0.01
  return lr


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_dataset, test_dataset = create_federated_dataset("cifar10", num_clients=FLAGS.total_num_clients)
  #train_dataset, test_dataset, warmup_dataset = create_federated_dataset("cifar10", num_clients=FLAGS.total_num_clients)
  federated_test_data = preprocess2(test_dataset.create_tf_dataset_from_all_clients(seed=42),FLAGS.batch_size)

  def tff_model_fn():
    """Constructs a fully initialized model for use in federated averaging."""
    #keras_model = create_original_fedavg_cnn_model(only_digits=True)
    keras_model = create_res_net((32,32,3),[2,2,2,2],10)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    return simple_fedavg_tf.KerasModelWrapper(keras_model,
                                              federated_test_data.element_spec, loss)

  iterative_process = simple_fedavg_tff.build_federated_averaging_process(
      tff_model_fn, lr_schedule, server_optimizer_fn, client_optimizer_fn)
  server_state = iterative_process.initialize()

  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  model = tff_model_fn()


  sampled_clients = train_dataset.client_ids[:FLAGS.num_clients]
  sampled_train_data = [
      preprocess(train_dataset.create_tf_dataset_for_client(client), FLAGS.batch_size)
      for client in sampled_clients
  ]

  # sampled_train_data = [
  #     preprocess(train_dataset.create_tf_dataset_from_all_clients(), FLAGS.batch_size)
  #     for client in sampled_clients
  # ]

  # sampled_warmup_data = [
  #     preprocess2(warmup_dataset.create_tf_dataset_for_client(client), FLAGS.batch_size)
  #     for client in sampled_clients
  # ]  




  model.from_weights(server_state.model_weights)
  accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, federated_test_data,
                                              metric)
  print(f'Pre-training validation accuracy: {accuracy * 100.0}')

  eval_sets = [federated_test_data for x in range(FLAGS.num_clients)]

  # server_state, train_metrics, eval_scores = iterative_process.next(
  #     server_state, sampled_warmup_data, eval_sets)
  # print(f'Warmup training loss: {train_metrics}')
  # for id, score in enumerate(eval_scores):
  #   print(f'Warmup client_id: {id} eval_score: {score}')
  # model.from_weights(server_state.model_weights)
  # accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, federated_test_data,
  #                                             metric)
  # print(f'Warmup validation accuracy: {accuracy * 100.0}')

  for round_num in range(FLAGS.total_rounds):
    server_state, train_metrics, eval_scores = iterative_process.next(
        server_state, sampled_train_data, eval_sets)
    print(f'Round {round_num} training loss: {train_metrics}')
    for id, score in enumerate(eval_scores):
      print(f'Round {round_num} client_id: {id} eval_score: {score}')
    if round_num % FLAGS.rounds_per_eval == 0:
      model.from_weights(server_state.model_weights)
      accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, federated_test_data,
                                                 metric)
      print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')


if __name__ == '__main__':
  app.run(main)