import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import custom_layers as clayers

def hr_5_3_0(inputs):
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[8],
        blocks=[3],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16],
        blocks=[3,3],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32],
        blocks=[3,3,3],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_3'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_4'
    )(x)
    x = clayers.HighResolutionFusion(
        filters=[8],
        name='Fusion_0'
    )(x)
    x = layers.Conv2D(
        8,
        2,
        strides=2,
        padding='same',
        name='Final_conv'
    )(x[0])
    x = layers.Conv2D(
        8,
        2,
        strides=2,
        padding='same',
        name='Final_conv'
    )(x)
    x = layers.Conv2D(
        8,
        2,
        strides=2,
        padding='same',
        name='Final_conv'
    )(x)
    x = layers.Flatten()(x)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs
