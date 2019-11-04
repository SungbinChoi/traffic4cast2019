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
num_sequence_per_day = num_frame_per_day - num_frame_sequence + 1      # 288-15+1=274
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
input_train_data_folder_path = '../0_data/' + target_city + '/' + target_city + '_training'
input_val_data_folder_path   = '../0_data/' + target_city + '/' + target_city + '_validation'
save_model_path = './models/1'
summary_path    = './summaries'
#
batch_size = 2
batch_size_val = 1
learning_rate = 3e-4
load_model_path = ''
is_training = True
num_epoch_to_train = 100000000
save_per_iteration = 2000
#
num_thread=4

def return_date(file_name):
    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date
  
  
def list_filenames(directory, excluded_dates):
    filenames = os.listdir(directory)
    np.random.shuffle(filenames)
    excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates]
    filenames = [x for x in filenames if return_date(x) not in excluded_dates]
    return filenames


def get_data_filepath_list(input_data_folder_path):
  
  data_filepath_list = []
  for filename in os.listdir(input_data_folder_path):
    if filename.split('.')[-1] != 'h5':     
      continue
    data_filepath_list.append(os.path.join(input_data_folder_path, filename))
  data_filepath_list = sorted(data_filepath_list)

  return data_filepath_list


def get_data(input_data_folder_path):
  
  data_filepath_list = []
  for filename in os.listdir(input_data_folder_path):
    if filename.split('.')[-1] != 'h5':     
      continue
    data_filepath_list.append(os.path.join(input_data_folder_path, filename))
  data_filepath_list = sorted(data_filepath_list)

  data_np_list = []
  for file_path in data_filepath_list:

    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = fr[a_group_key]
    assert data.shape[0] == num_frame_per_day
    data_np_list.append(data)

  return data_np_list
  




if __name__ == '__main__':
  
  random.seed(SEED)
  np.random.seed(SEED)
  tf.set_random_seed(SEED)
  
  trainer = Trainer(height, width, visual_input_channels, visual_output_channels, vector_input_channels, learning_rate, 
                    save_model_path, load_model_path, summary_path, is_training)
  tf.reset_default_graph() 
  
  
  
  try:
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
  except Exception:
            print('save_model_path not made')
            exit(-1)
            
            
  val_data   = get_data(input_val_data_folder_path)
  print('val_data\t',   len(val_data),  '\t', val_data[0].shape,   '\tsize:', sys.getsizeof(val_data))
  
  train_data_filepath_list = get_data_filepath_list(input_train_data_folder_path)
  print('train_data_filepath_list\t', len(train_data_filepath_list),)
  


  train_set = []
  #
  for i in range(len(train_data_filepath_list)):
    for j in range(num_sequence_per_day):
      train_set.append( (i,j) )
  num_iteration_per_epoch = int(len(train_set) / batch_size)
  print('num_iteration_per_epoch:', num_iteration_per_epoch)
  
  
  val_set = []
  #
  for i in range(len(val_data)):
    for j in range(0, num_sequence_per_day, num_frame_sequence):
      val_set.append( (i,j) )
  num_val_iteration_per_epoch = int(len(val_set) / batch_size_val)    
  print('num_val_iteration_per_epoch:', num_val_iteration_per_epoch)  
  
  
  
  

  
  
  
  
    
    
    
  
  
  train_input_queue  = queue.Queue()
  train_output_queue = queue.Queue()
  
  def load_train_multithread():
    
    while True:
      if train_input_queue.empty() or train_output_queue.qsize() > 8:
        time.sleep(0.1)
        continue
      i_j_list = train_input_queue.get()
      
      
      train_orig_data_batch_list  = []
      train_data_batch_list = []  
      train_data_mask_list = []   
      train_data_time_list  = []
      train_stat_batch_list = [] 
      for train_i_j in i_j_list:
          (i,j) = train_i_j
        
          file_path = train_data_filepath_list[i]
          
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = fr[a_group_key]
          assert data.shape[0] == num_frame_per_day

          train_data_time_list.append(float(j)/float(num_frame_per_day))

          train_orig_data_batch_list.append(data[j:j+num_frame_sequence,:,:,:][np.newaxis,:,:,:,:])
          

          data_sliced = data[j:j+num_frame_sequence,:,:,:]
          #
          data_mask = (np.max(data_sliced, axis=3) == 0)
          train_data_mask_list.append(data_mask[np.newaxis,:,:,:])
          
          data_direction = np.zeros((num_frame_sequence, height, width, 4),np.uint8)
          if 1:
            c=2
            data_direction[data_sliced[:,:,:,c]==n,0] = 1
            data_direction[data_sliced[:,:,:,c]==s,1] = 1
            data_direction[data_sliced[:,:,:,c]==e,2] = 1
            data_direction[data_sliced[:,:,:,c]==w,3] = 1

          train_data_batch_list.append(np.concatenate([data_sliced[:,:,:,0:2], data_direction], axis=-1)[np.newaxis,:,:,:,:])
          
      train_data_time_list = np.asarray(train_data_time_list)
      input_time = np.reshape(train_data_time_list, (batch_size, 1))


      train_orig_data_batch  = np.concatenate(train_orig_data_batch_list,  axis=0)
      train_data_batch = np.concatenate(train_data_batch_list, axis=0)
      train_data_mask  = np.concatenate(train_data_mask_list, axis=0)
      
      
      input_data = train_data_batch[:,:num_frame_before ,:,:,:]
      true_label = train_data_batch[:, num_frame_before:,:,:,:]
      orig_label = train_orig_data_batch[:, num_frame_before:,:,:,:]
      
      input_data = input_data.astype(np.float32)
      true_label = true_label.astype(np.float32)
      orig_label = orig_label.astype(np.float32)
      
      input_data[:,:,:,:,0:2] = input_data[:,:,:,:,0:2]/255.0
      true_label[:,:,:,:,0:2] = true_label[:,:,:,:,0:2]/255.0
      orig_label = orig_label / 255.0
      
      input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size, height, width, -1))  
      true_label = np.moveaxis(true_label, 1, -1).reshape((batch_size, height, width, -1))  
      orig_label = np.moveaxis(orig_label, 1, -1).reshape((batch_size, height, width, -1))  

      orig_label_mask = np.ones((batch_size, num_frame_sequence-num_frame_before, height, width, num_channel), np.float32)
      orig_label_mask[train_data_mask[:,num_frame_before: ,:,:], :] = 0.0  
      orig_label_mask = np.moveaxis(orig_label_mask, 1, -1).reshape((batch_size, height, width, -1))

      input_data_mask = np.zeros((batch_size, num_frame_before, height, width, num_channel_discretized), np.bool)
      input_data_mask[train_data_mask[:,:num_frame_before ,:,:], :] = True
      input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape((batch_size, height, width, -1))
      input_data[input_data_mask] = -1.0                                  

      train_output_queue.put( (input_data, orig_label, orig_label_mask, input_time) )



  thread_list = []
  assert num_thread > 0
  for i in range(num_thread):
    t = threading.Thread(target=load_train_multithread)
    t.start()

  
  
  
  global_step = 0
  for epoch in range(num_epoch_to_train):
    
    np.random.shuffle(train_set)

    for a in range(num_iteration_per_epoch):
      
      i_j_list = []      
      for train_i_j in train_set[a * batch_size : (a+1) * batch_size]:
        i_j_list.append(train_i_j)
      train_input_queue.put(i_j_list)
    
    
    for a in range(num_iteration_per_epoch):
      
      while train_output_queue.empty():
        time.sleep(0.1)
      (input_data, true_label, true_label_mask, input_time) = train_output_queue.get()
      run_out = trainer.update(input_data, true_label, true_label_mask, input_time)
      global_step += 1

      if global_step % save_per_iteration == 0:
      
        eval_loss_list = list()
        for a in range(num_val_iteration_per_epoch):
        
          val_orig_data_batch_list  = []
          val_data_batch_list = []   
          val_data_mask_list = [] 
          val_data_time_list  = []
          val_stat_batch_list = []   
          for i_j in val_set[a * batch_size_val : (a+1) * batch_size_val]:
            (i,j) = i_j
            val_data_time_list.append(float(j)/float(num_frame_per_day))
            val_orig_data_batch_list.append(val_data[i][j:j+num_frame_sequence,:,:,:][np.newaxis,:,:,:,:])
            data_sliced = val_data[i][j:j+num_frame_sequence,:,:,:]
            data_mask = (np.max(data_sliced, axis=3) == 0)
            val_data_mask_list.append(data_mask[np.newaxis,:,:,:])
            data_direction = np.zeros((num_frame_sequence, height, width, 4),np.uint8)
            if 1:
              c=2
              data_direction[data_sliced[:,:,:,c]==n,0] = 1
              data_direction[data_sliced[:,:,:,c]==s,1] = 1
              data_direction[data_sliced[:,:,:,c]==e,2] = 1
              data_direction[data_sliced[:,:,:,c]==w,3] = 1
            val_data_batch_list.append(np.concatenate([data_sliced[:,:,:,0:2], data_direction], axis=-1)[np.newaxis,:,:,:,:])

          val_data_time_list = np.asarray(val_data_time_list)
          input_time = np.reshape(val_data_time_list, (batch_size_val, 1))

          val_orig_data_batch  = np.concatenate(val_orig_data_batch_list,  axis=0)
          val_data_batch = np.concatenate(val_data_batch_list, axis=0)
          val_data_mask = np.concatenate(val_data_mask_list, axis=0)  
          
          input_data = val_data_batch[:,:num_frame_before ,:,:,:] 
          true_label = val_data_batch[:, num_frame_before:,:,:,:]  
          orig_label = val_orig_data_batch[:, num_frame_before:,:,:,:]

          input_data = input_data.astype(np.float32)
          true_label = true_label.astype(np.float32)
          orig_label = orig_label.astype(np.float32)

          input_data[:,:,:,:,0:2] = input_data[:,:,:,:,0:2]/255.0
          true_label[:,:,:,:,0:2] = true_label[:,:,:,:,0:2]/255.0
          orig_label = orig_label / 255.0
          
          
          input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size_val, height, width, -1))  
          true_label = np.moveaxis(true_label, 1, -1).reshape((batch_size_val, height, width, -1))  
          orig_label = np.moveaxis(orig_label, 1, -1).reshape((batch_size_val, height, width, -1))  

          orig_label_mask = np.ones((batch_size_val, num_frame_sequence-num_frame_before, height, width, num_channel), np.float32)
          orig_label_mask[val_data_mask[:,num_frame_before: ,:,:], :] = 0.0  
          orig_label_mask = np.moveaxis(orig_label_mask, 1, -1).reshape((batch_size_val, height, width, -1))

          input_data_mask = np.zeros((batch_size_val, num_frame_before, height, width, num_channel_discretized), np.bool)
          input_data_mask[val_data_mask[:,:num_frame_before ,:,:], :] = True
          input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape((batch_size_val, height, width, -1))
          input_data[input_data_mask] = -1.0                                  
          
          run_out = trainer.evaluate(input_data, orig_label, orig_label_mask, input_time)
          eval_loss_list.append(run_out['loss'])

        print('global_step:', global_step, '\t', 'epoch:', epoch, '\t', 'eval_loss:', np.mean(eval_loss_list))

        trainer.save_model(global_step)
        trainer.write_summary(global_step)
      
