from train import fit
from return_train_test_dataset import the_train_data_provider
from return_train_test_dataset import the_test_data_provider

BUFFER_SIZE = 1000
BATCH_SIZE = 1
steps = 5000
train_path = '/content/drive/MyDrive/data_sat_to_map/maps/train/train/'
test_path = '/content/drive/MyDrive/data_sat_to_map/maps/val/val/'
train_ds = the_train_data_provider(train_path,BUFFER_SIZE, BATCH_SIZE)
test_ds = the_test_data_provider(test_path, BUFFER_SIZE, BATCH_SIZE)


fit(train_ds, test_ds, steps= 50000)