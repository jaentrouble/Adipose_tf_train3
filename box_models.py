import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dense_512(encode_input, pos_input):
    x = layers.Concatenate()(encode_input,pos_input)
    x = layers.Dense(512)
    x = layers.Dense(4)
    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs