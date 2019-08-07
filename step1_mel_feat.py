import os
from get_data_mel import get_data_mel

data_train_dir  =  "./test_file/"
#data_test_dir   =  "./test_data/"

store_dir       = "./entire_spec"
store_train_dir = "./entire_spec/test_spec"
#store_test_dir  = "./../11_11_mel_spec/data_test"

if not os.path.exists(store_dir):
    os.makedirs(store_dir)

if not os.path.exists(store_train_dir):
    os.makedirs(store_train_dir)

#if not os.path.exists(store_test_dir):
#    os.makedirs(store_test_dir)

#For Training 
get_data_mel(data_train_dir, store_train_dir, 1)

#For Testing
#get_data_mel(data_test_dir, store_test_dir, 0)
