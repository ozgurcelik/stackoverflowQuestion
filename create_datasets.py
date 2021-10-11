import collections
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff

np.random.seed(42)
tf.random.set_seed(42)

def create_federated_dataset(dataset_name, num_clients = 10):
    train_images, train_labels = tfds.as_numpy(
        tfds.load(
            name=dataset_name,
            split='train',
            batch_size=-1,
            as_supervised=True,
        ))

    test_images, test_labels = tfds.as_numpy(
        tfds.load(
            name=dataset_name,
            split='test',
            batch_size=-1,
            as_supervised=True,
        ))

    if dataset_name=="cifar10":
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
    else:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = (train_images-mean)/std
    test_images = (test_images-mean)/std

    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    #tf.print(indices)

    train_images = train_images[indices]
    train_labels = train_labels[indices]
 
    total_image_count_train = len(train_labels)
    total_image_count_test = len(test_labels)
    image_per_set_train = int(np.floor(total_image_count_train/num_clients))
    image_per_set_test = int(np.floor(total_image_count_test/num_clients))

    client_train_dataset = collections.OrderedDict()
    client_test_dataset = collections.OrderedDict()
    #client_warmup_dataset = collections.OrderedDict()
    for i in range(0, num_clients):
        client_name = str(i)
        start_train = image_per_set_train * i
        end_train = image_per_set_train * (i+1)
        start_test = image_per_set_test * i
        end_test = image_per_set_test * (i+1)

        data_train = collections.OrderedDict((('label', train_labels[start_train:end_train]), ('pixels', train_images[start_train:end_train])))
        #data_train = collections.OrderedDict((('label', train_labels), ('pixels', train_images)))
        data_test = collections.OrderedDict((('label', test_labels[start_test:end_test]), ('pixels', test_images[start_test:end_test])))
        #data_warmup = collections.OrderedDict((('label', train_labels[i:i+1]), ('pixels', train_images[i:i+1])))
        client_train_dataset[client_name] = data_train
        client_test_dataset[client_name] = data_test
        #client_warmup_dataset[client_name] = data_warmup

    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    test_dataset = tff.simulation.FromTensorSlicesClientData(client_test_dataset)
    #warmup_dataset = tff.simulation.FromTensorSlicesClientData(client_warmup_dataset)

    return train_dataset, test_dataset


def preprocess(dataset, batch_size=20):

    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.image.random_flip_left_right(element['pixels']),
            y=element['label'])

    return dataset.batch(batch_size).map(batch_format_fn)

def preprocess_repeat(dataset, batch_size=20, repeat_count=1):

    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.image.random_flip_left_right(element['pixels']),
            y=element['label'])

    return dataset.repeat(repeat_count).shuffle(100, seed=42).batch(batch_size).map(batch_format_fn)



def preprocess2(dataset, batch_size=20):

    def batch_format_fn(element):
        return collections.OrderedDict(
            x=element['pixels'],
            y=element['label'])

    return dataset.batch(batch_size).map(batch_format_fn)