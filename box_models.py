import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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