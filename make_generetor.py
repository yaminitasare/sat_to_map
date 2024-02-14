import tensorflow as tf
from IPython import display
from tensorflow.keras import layers

def generator():
    # Encoder
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    conv1 = layers.Conv2D(64, (3, 3), padding='same', strides = (2,2))(inputs)
    conv1 = layers.LeakyReLU()(conv1)
    # add skip
    conv2 = layers.Conv2D(128, (3,3), padding = 'same', strides = (2,2))(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    # add skip
    conv3 = layers.Conv2D(256, (3,3), padding = 'same', strides = (2,2))(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    # add skip
    conv4 = layers.Conv2D(512, (3,3), padding = 'same', strides = (2,2))(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU()(conv4)
    #add skip
    conv5 = layers.Conv2D(512, (3,3), padding = 'same', strides = (2,2))(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU()(conv5)
    #add skip
    conv6 = layers.Conv2D(512, (3,3), padding = 'same', strides = (2,2))(conv5)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.LeakyReLU()(conv6)
    #add skip
    conv7 = layers.Conv2D(512, (3,3), padding = 'same', strides = (2,2))(conv6)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.LeakyReLU()(conv7)
    #add skip
    conv8 = layers.Conv2D(512, (3,3), padding = 'same', strides = (2,2))(conv7)
    conv8 = tf.keras.layers.Activation('relu')(conv8)
    ## reversal starts here.
    # add conv2d transpose.
    conv_trans = layers.Conv2DTranspose(512, (3,3), padding = 'same', strides = (2,2))(conv8)
    conv_trans = layers.BatchNormalization()(conv_trans)
    conv_trans = layers.Dropout(0.4)(conv_trans)
    merge1 = layers.concatenate([conv7, conv_trans], axis=3)
    merge1 = tf.keras.layers.Activation('relu')(merge1)
    #2nd trns layer
    conv_trans2 = layers.Conv2DTranspose(512, (3,3), padding = 'same', strides = (2,2))(merge1)
    conv_trans2 = layers.BatchNormalization()(conv_trans2)
    conv_trans2 = layers.Dropout(0.4)(conv_trans2)
    merge2 = layers.concatenate([conv6, conv_trans2], axis=3)
    merge2 = tf.keras.layers.Activation('relu')(merge2)
    #3rd trns layer
    conv_trans3 = layers.Conv2DTranspose(512, (3,3), padding = 'same', strides = (2,2))(merge2)
    conv_trans3 = layers.BatchNormalization()(conv_trans3)
    conv_trans3 = layers.Dropout(0.4)(conv_trans3)
    merge3 = layers.concatenate([conv5, conv_trans3], axis=3)
    merge3 = tf.keras.layers.Activation('relu')(merge3)
    #4th trns layer
    conv_trans4 = layers.Conv2DTranspose(512, (3,3), padding = 'same', strides = (2,2))(merge3)
    conv_trans4 = layers.BatchNormalization()(conv_trans4)
    merge4 = layers.concatenate([conv4, conv_trans4], axis=3)
    merge4 = tf.keras.layers.Activation('relu')(merge4)
    # fifth trns layer
    conv_trans5 = layers.Conv2DTranspose(256, (3,3), padding = 'same', strides = (2,2))(merge4)
    conv_trans5 = layers.BatchNormalization()(conv_trans5)
    merge5 = layers.concatenate([conv3, conv_trans5], axis=3)
    merge5 = tf.keras.layers.Activation('relu')(merge5)
    # sixth trns layer
    conv_trans6 = layers.Conv2DTranspose(128, (3,3), padding = 'same', strides = (2,2))(merge5)
    conv_trans6 = layers.BatchNormalization()(conv_trans6)
    merge6 = layers.concatenate([conv2, conv_trans6], axis=3)
    merge6 = tf.keras.layers.Activation('relu')(merge6)
    # seventh trns layer
    conv_trans7 = layers.Conv2DTranspose(64, (3,3), padding = 'same', strides = (2,2))(merge6)
    conv_trans7 = layers.BatchNormalization()(conv_trans7)
    merge7 = layers.concatenate([conv1, conv_trans7], axis=3)
    merge7 = tf.keras.layers.Activation('relu')(merge7)
    # final
    conv_trans8 = layers.Conv2DTranspose(3, (3,3), padding = 'same', strides = (2,2))(merge7)
    conv_trans8 = tf.keras.layers.Activation('relu')(conv_trans8)
    return tf.keras.Model(inputs, conv_trans8)


def my_generator():
    generator = generator()
    return generator


# LAMBDA = 100
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss