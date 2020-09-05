import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dense_4(encode_input, pos_input):
    x = layers.Concatenate()([encode_input,pos_input])
    x = layers.Dense(4)(x)
    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs