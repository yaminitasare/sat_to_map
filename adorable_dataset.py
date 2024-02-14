import tensorflow as tf

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)


    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return real_image,input_image