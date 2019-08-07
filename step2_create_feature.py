import os
from get_test_data import get_test_data
#from get_train_data import get_train_data
#from get_pre_train_data import get_pre_train_data

# For saving data
store_dir           ='./split_spec'
#save_train_dir     = './data/data_train/'
save_test_dir      = './split_spec/test_split_spec/'
#save_pre_train_dir = './data/data_pre_train/'

if not os.path.exists(store_dir):
    os.makedirs(store_dir)

#if not os.path.exists(save_train_dir):
#    os.makedirs(save_train_dir)

if not os.path.exists(save_test_dir):
    os.makedirs(save_test_dir)

#if not os.path.exists(save_pre_train_dir):
#    os.makedirs(save_pre_train_dir)


# For input data
#data_train_dir      =  './../12_11_group_mel_spec/data_train/'
data_test_dir       =  './entire_spec/test_spec/'   #no group for test data
#data_pre_train_dir  =  './../11_11_mel_spec/data_train/'   #no group for test data

#for training data
#print('================================== Creating Training File\n')
#get_train_data(save_train_dir,
#               data_train_dir
#              )

#for testing data
print('================================== Creating Testing File\n')
get_test_data(save_test_dir,
               data_test_dir
              )

#for pre train data
#print('================================== Creating Pre_Train File\n')
#get_pre_train_data(save_pre_train_dir,
#                   data_pre_train_dir
#                  )
