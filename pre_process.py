import tensorflow as tf
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers
import matplotlib.pyplot as plt
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

        
def resize(sat_img, map_img, height, width):
    sat_img = tf.image.resize(sat_img, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    map_img = tf.image.resize(map_img, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return sat_img, map_img

def random_crop_data(sat_img, map_img):
    stacked_image = tf.stack([sat_img, map_img], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]    

def normalize(sat_img, map_img):
    sat_img = (sat_img / 127.5) - 1
    map_img = (map_img / 127.5) - 1
    return sat_img, map_img

@staticmethod
@tf.function()
def jitter(sat_img, map_img):
    sat_img, map_img = resize(sat_img, map_img, 286, 286)
    sat_img, map_img = random_crop_data(sat_img, map_img)
    if tf.random.uniform(()) > 0.5:
        sat_img = tf.image.flip_left_right(sat_img)
        map_img = tf.image.flip_left_right(map_img)
    return sat_img, map_img