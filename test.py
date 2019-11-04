import random
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import datetime
import time
import queue
import threading
import logging
from PIL import Image
import itertools
import yaml
import re
import os
import glob
import shutil
import sys
import copy
import h5py
from net_all import *
from trainer_all import *



SEED = 0
num_train_file = 285
num_frame_per_day = 288
num_frame_before = 12
num_frame_sequence = 15
num_sequence_per_day = num_frame_per_day - num_frame_sequence + 1
height=495
width =436
num_channel=3
num_channel_discretized = 6 # 2 + 4
visual_input_channels=72   #  12*(2+4)
visual_output_channels=9   #  3*3 
vector_input_channels=1    #  start time point
#
n = 1
s = 255
e = 85
w = 170
tv = 16
#
target_city = 'Berlin'
test_start_index_list = np.array([ 18,  57, 114, 174, 222], np.int32)    # 'Berlin' 
#test_start_index_list = np.array([ 45, 102, 162, 210, 246], np.int32)   # 'Moscow' # 'Istanbul'
input_train_data_folder_path = '../0_data/' + target_city + '/' + target_city + '_training'
input_val_data_folder_path   = '../0_data/' + target_city + '/' + target_city + '_validation'
input_test_data_folder_path  = '../0_data/' + target_city + '/' + target_city + '_test'
save_model_path = './models/1'
summary_path    = './summaries'
#
batch_size_test = 5
learning_rate = 3e-4
load_model_path = 'trained_models/' + target_city 
is_training = False                               



def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()
    

def get_data_filepath_list(input_data_folder_path):
  
  data_filepath_list = []
  for filename in os.listdir(input_data_folder_path):
    if filename.split('.')[-1] != 'h5':     
      continue
    data_filepath_list.append(os.path.join(input_data_folder_path, filename))
  data_filepath_list = sorted(data_filepath_list)

  return data_filepath_list



if __name__ == '__main__':
  
  random.seed(SEED)
  np.random.seed(SEED)
  tf.set_random_seed(SEED)
  
  trainer = Trainer(height, width, visual_input_channels, visual_output_channels, vector_input_channels, learning_rate, 
                    save_model_path, load_model_path, summary_path, is_training)
  tf.reset_default_graph() 

  test_data_filepath_list = get_data_filepath_list(input_test_data_folder_path)
  print('test_data_filepath_list\t', len(test_data_filepath_list),)
  
  test_output_filepath_list = list()
  for test_data_filepath in test_data_filepath_list:
    filename = test_data_filepath.split('/')[-1]
    test_output_filepath_list.append('output/' + target_city + '/' + target_city + '_test' + '/' + filename)

  try:
    if not os.path.exists('output'):
      os.makedirs('output')
    os.makedirs('output/' + target_city)
    os.makedirs('output/' + target_city + '/' + target_city + '_test')
  except Exception:
    print('output path not made')
    exit(-1)  


  for i in range(len(test_data_filepath_list)):
    file_path = test_data_filepath_list[i]
    out_file_path = test_output_filepath_list[i]

    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = fr[a_group_key]
    assert data.shape[0] == num_frame_per_day
    data = np.array(data, np.uint8)      

    test_data_batch_list  = []  
    test_data_time_list   = []
    test_data_mask_list = []


    for j in test_start_index_list:

      test_data_time_list.append(float(j)/float(num_frame_per_day))

      data_sliced = data[j:j+num_frame_before,:,:,:]
      #
      data_mask = (np.max(data_sliced, axis=3) == 0)
      test_data_mask_list.append(data_mask[np.newaxis,:,:,:])
      data_direction = np.zeros((num_frame_before, height, width, 4),np.uint8)
      if 1:
            c=2
            data_direction[data_sliced[:,:,:,c]==n,0] = 1
            data_direction[data_sliced[:,:,:,c]==s,1] = 1
            data_direction[data_sliced[:,:,:,c]==e,2] = 1
            data_direction[data_sliced[:,:,:,c]==w,3] = 1
            test_data_batch_list.append(np.concatenate([data_sliced[:,:,:,0:2], data_direction], axis=-1)[np.newaxis,:,:,:,:])


    test_data_time_list = np.asarray(test_data_time_list, np.float32)
    input_time = np.reshape(test_data_time_list, (batch_size_test, 1)) 

    test_data_mask = np.concatenate(test_data_mask_list, axis=0)
    

    input_data = np.concatenate(test_data_batch_list, axis=0).astype(np.float32)  
    input_data[:,:,:,:,0:2] = input_data[:,:,:,:,0:2]/255.0
    input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size_test, height, width, -1))

    input_data_mask = np.zeros((batch_size_test, num_frame_before, height, width, num_channel_discretized), np.bool)
    input_data_mask[test_data_mask[:,:num_frame_before ,:,:], :] = True
    input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape((batch_size_test, height, width, -1))
    input_data[input_data_mask] = -1.0                                  

    true_label_mask = np.ones((batch_size_test, height, width, visual_output_channels), dtype=np.float32)

    prediction_list = []
    for b in range(batch_size_test):
      run_out_one = trainer.infer(input_data[b,:,:,:][np.newaxis,:,:,:], 
                              input_time[b,:][np.newaxis,:], 
                              true_label_mask[b,:,:,:][np.newaxis,:,:,:])   
      prediction_one = run_out_one['predict']
      prediction_list.append(prediction_one)
    prediction = np.concatenate(prediction_list, axis=0)    
    prediction = np.moveaxis(np.reshape(prediction, (batch_size_test, height, width, num_channel, num_frame_sequence-num_frame_before, )), -1, 1)
    prediction = prediction.astype(np.float32) * 255.0
    prediction = np.rint(prediction)
    prediction = np.clip(prediction, 0.0, 255.0).astype(np.uint8)
    assert prediction.shape == (batch_size_test, num_frame_sequence-num_frame_before, height, width, num_channel)

    write_data(prediction, out_file_path)


    
    
    