import tensorflow as tf
from IPython import display
from tensorflow.keras import layers

def ASPP(inputs, output_stride):
    conv_filters = 256
    aspp_conv_0 = tf.keras.layers.Conv2D(conv_filters, (1, 1), padding='same', use_bias=False,
                                         kernel_initializer='he_normal', name='aspp0')(inputs)
    aspp_conv_0 = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='aspp0_BN')(aspp_conv_0)
    aspp_conv_0 = tf.keras.layers.Activation('relu', name='aspp0_activation')(aspp_conv_0)

    aspp_conv_3 = tf.keras.layers.Conv2D(conv_filters, (3, 3), dilation_rate=(6, 6), padding='same',
                                         use_bias=False, kernel_initializer='he_normal', name='aspp3')(inputs)
    aspp_conv_3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='aspp3_BN')(aspp_conv_3)
    aspp_conv_3 = tf.keras.layers.Activation('relu', name='aspp3_activation')(aspp_conv_3)

    aspp_conv_6 = tf.keras.layers.Conv2D(conv_filters, (3, 3), dilation_rate=(12, 12), padding='same',
                                         use_bias=False, kernel_initializer='he_normal', name='aspp6')(inputs)
    aspp_conv_6 = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='aspp6_BN')(aspp_conv_6)
    aspp_conv_6 = tf.keras.layers.Activation('relu', name='aspp6_activation')(aspp_conv_6)

    aspp_pooling = tf.keras.layers.MaxPooling2D(pool_size=(output_stride, output_stride), strides=(1, 1),
                                                padding='same', name='aspp_pool')(inputs)
    aspp_pooling = tf.keras.layers.Conv2D(conv_filters, (1, 1), padding='same', use_bias=False,
                                          kernel_initializer='he_normal', name='aspp_pool_conv')(aspp_pooling)
    aspp_pooling = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='aspp_pool_conv_BN')(aspp_pooling)
    aspp_pooling = tf.keras.layers.Activation('relu', name='aspp_pool_conv_activation')(aspp_pooling)

    aspp_output = tf.keras.layers.Concatenate(axis=-1, name='aspp_concat')([aspp_conv_0, aspp_conv_3, aspp_conv_6, aspp_pooling])

    return aspp_output

def dynamic_conv(inputs, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1)):
    padding = 'same'
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding,
                               strides=strides, dilation_rate=dilation_rate,
                               kernel_initializer=kernel_initializer, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def generator():
    # Encoder
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    conv1 = dynamic_conv(inputs, 64, (3, 3), strides=(2, 2))
    # add skip
    conv2 = dynamic_conv(conv1, 128, (3, 3), strides=(2, 2))
    # add skip
    conv3 = dynamic_conv(conv2, 256, (3, 3), strides=(2, 2))
    # add skip
    conv4 = dynamic_conv(conv3, 512, (3, 3), strides=(2, 2))
    #add skip
    conv5 = dynamic_conv(conv4, 512, (3, 3), strides=(2, 2))
    #add skip
    conv6 = dynamic_conv(conv5, 512, (3, 3), strides=(2, 2))
    #add skip
    conv7 = dynamic_conv(conv6, 512, (3, 3), strides=(2, 2))
    #add skip
    conv8 = dynamic_conv(conv7, 512, (3, 3), strides=(2, 2))

    # ASPP module
    aspp_out = ASPP(conv8, output_stride=16)

    ## reversal starts here.
    # add dynamic dilated conv2d transpose.
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(aspp_out)
    conv_trans = dynamic_conv(up1, 512, (3, 3), dilation_rate=(2, 2))
    merge1 = tf.keras.layers.concatenate([conv7, conv_trans], axis=3)
    merge1 = tf.keras.layers.Activation('relu')(merge1)
    #2nd trns layer
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(merge1)
    conv_trans2 = dynamic_conv(up2, 512, (3, 3), dilation_rate=(2, 2))
    merge2 = tf.keras.layers.concatenate([conv6, conv_trans2], axis=3)
    merge2 = tf.keras.layers.Activation('relu')(merge2)
    #3rd trns layer
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(merge2)
    conv_trans3 = dynamic_conv(up3, 512, (3, 3), dilation_rate=(2, 2))
    merge3 = tf.keras.layers.concatenate([conv5, conv_trans3], axis=3)
    merge3 = tf.keras.layers.Activation('relu')(merge3)
    #4th trns layer
    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(merge3)
    conv_trans4 = dynamic_conv(up4, 512, (3, 3), dilation_rate=(2, 2))
    merge4 = tf.keras.layers.concatenate([conv4, conv_trans4], axis=3)
    merge4 = tf.keras.layers.Activation('relu')(merge4)
    # fifth trns layer
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(merge4)
    conv_trans5 = dynamic_conv(up5, 256, (3, 3), dilation_rate=(2, 2))
    merge5 = tf.keras.layers.concatenate([conv3, conv_trans5], axis=3)
    merge5 = tf.keras.layers.Activation('relu')(merge5)
    # sixth trns layer
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(merge5)
    conv_trans6 = dynamic_conv(up6, 128, (3, 3), dilation_rate=(2, 2))
    merge6 = tf.keras.layers.concatenate([conv2, conv_trans6], axis=3)
    merge6 = tf.keras.layers.Activation('relu')(merge6)
    # seventh trns layer
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(merge6)
    conv_trans7 = dynamic_conv(up7, 64, (3, 3), dilation_rate=(2, 2))
    merge7 = tf.keras.layers.concatenate([conv1, conv_trans7], axis=3)
    merge7 = tf.keras.layers.Activation('relu')(merge7)
    # final
    conv_trans8 = tf.keras.layers.Conv2DTranspose(3, (3,3), padding = 'same', strides = (2,2))(merge7)
    conv_trans8 = tf.keras.layers.Activation('relu')(conv_trans8)

    return tf.keras.Model(inputs, conv_trans8)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
 gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
 l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
 total_gen_loss = gan_loss + (LAMBDA * l1_loss)
 return total_gen_loss, gan_loss, l1_loss
