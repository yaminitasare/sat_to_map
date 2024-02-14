from adorable_dataset import load
from pre_process import jitter
from pre_process import resize
from pre_process import normalize
IMG_WIDTH = 256
IMG_HEIGHT = 256
def load_train_data(image_file):
    sat_img, map_img = load(image_file)
    sat_img, map_img = jitter(sat_img, map_img)
    sat_img, map_img = normalize(sat_img, map_img)
    return sat_img, map_img

def load_test_data(image_file):
    sat_img, map_img = load(image_file)
    sat_img, map_img = resize(sat_img, map_img, IMG_HEIGHT, IMG_WIDTH)
    sat_img, map_img = normalize(sat_img, map_img)
    return sat_img, map_img