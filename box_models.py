import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""Models to predict box

NOTE: Here, argument 'image_size' is (WIDTH, HEIGHT)
"""

def dense_4(encode_input, pos_input, *args, **kwargs):
    x = layers.Concatenate()([encode_input,pos_input])
    x = layers.Dense(4)(x)
    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs

def dense_128_4_norm(encode_input, pos_input, image_size):
    x = layers.Concatenate()([encode_input, pos_input])
    x = layers.Dense(128, activation='relu')(x)

    # This x is normalized value (for stability)
    x = layers.Dense(4)(x)
    
    output_ratio = tf.reshape(tf.convert_to_tensor([
        image_size[0],
        image_size[1],
        image_size[0],
        image_size[1]
    ],dtype=tf.float32),[1,4])

    x = layers.Multiply()([x, output_ratio])
    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs

def dense_128_4_normoutpos(encode_input, pos_input, image_size):
    # Normalize position for stability
    width = image_size[0]
    height = image_size[1]
    ratio = tf.convert_to_tensor([1/width,1/height], dtype=tf.float32)
    ratio = tf.reshape(ratio, [1,2])
    pos_input = layers.Multiply()([pos_input, ratio])

    x = layers.Concatenate()([encode_input, pos_input])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(4)(x)

    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs

def dense_128_4(encode_input, pos_input):
    # Takes NORMALIZED position input
    x = layers.Concatenate()([encode_input, pos_input])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(4)(x)

    outputs = layers.Activation('linear',dtype='float32')(x)
    return outputs
