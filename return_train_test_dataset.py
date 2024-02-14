
from make_dataset import load_train_data
from make_dataset import load_test_data
import tensorflow as tf
train_path = '/content/drive/MyDrive/data_sat_to_map/maps/train/train/'
test_path = '/content/drive/MyDrive/data_sat_to_map/maps/val/val/'
def the_train_data_provider(train_path,BUFFER_SIZE, BATCH_SIZE):
    train_dataset = tf.data.Dataset.list_files(str(train_path +  '*.jpg'))
    train_dataset = train_dataset.map(load_train_data,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    return train_dataset
def the_test_data_provider(test_path, BUFFER_SIZE, BATCH_SIZE):
    test_dataset = tf.data.Dataset.list_files(str(test_path + '*.jpg'))
    test_dataset = test_dataset.map(load_test_data)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset
