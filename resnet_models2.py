import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow_addons.layers.normalizations as tfa_norms

tf.random.set_seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

def norm(input_tensor, norm='group1'):
  if norm == 'group':
    norm_out = tfa_norms.GroupNormalization()(input_tensor)
  else:
    norm_out = BatchNormalization(epsilon=1e-05, momentum=0.9)(input_tensor)
    #norm_out = BatchNormalization()(input_tensor)
  return norm_out

def BasicBlock(x, filters: int, downsample: bool):
    y = Conv2D(filters=filters,
               kernel_size=3,
               strides=(1 if not downsample else 2),
               padding="same",
               use_bias="False",
               kernel_initializer=initializer)(x)
    y = norm(y)
    y = ReLU()(y)
    y = Conv2D(filters=filters,
               kernel_size=3,
               strides=1,
               padding="same",
               use_bias="False",
               kernel_initializer=initializer)(y)
    y = norm(y)

    if downsample:
        x = Conv2D(filters=filters,
               kernel_size=1,
               strides=2,
               padding="valid",
               use_bias="False",
               kernel_initializer=initializer)(x)
        x = norm(x)
    
    out = Add()([x, y])
    out = ReLU()(out)

    return out

def create_res_net(input_shape, num_blocks_list, num_classes):
    inputs = Input(shape=input_shape)
    num_filters = 64
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same",
               kernel_initializer=initializer)(inputs)
    t = norm(t)
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = BasicBlock(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(t)

    model = Model(inputs, outputs)

    return model