from random import shuffle
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
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers
from net_all import *


GPU='0'
logger = logging.getLogger("traffic")




class Trainer(object):

  def __init__(self, height, width, visual_input_channels, visual_output_channels, vector_input_channels, learning_rate,
                     model_path, load_model_path, summary_path, is_training):

    self.model_path = model_path
    self.load_model_path = load_model_path
    self.summary_path = summary_path
    self.is_training = is_training
    self.keep_checkpoints = 100000000
    
    self.possible_output_nodes = ['predict', ] 
    
    self.graph = tf.Graph()
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config, graph=self.graph)
    self.saver = None
    self.step = 0
    
    self.stats = {'train_loss': [], 'eval_loss': [], }
    self.stats_aux = {}
    self.summary_writer = tf.summary.FileWriter(self.summary_path)

    with tf.device('/gpu:' + GPU):

      with self.graph.as_default():
        self.model = NetA(height, width, visual_input_channels, visual_output_channels, vector_input_channels, learning_rate)
        self.model.build()
        
      if load_model_path != '':
        self._load_graph()
      else:
        self._initialize_graph()

      self.inference_dict = {'predict': self.model.predict,
                            }
      self.update_dict    = {'loss':   self.model.loss,
                             'update_batch': self.model.update_batch
                            }
      self.evaluate_dict = {'predict': self.model.predict,
                            'loss':   self.model.loss,
                            }
 
    
  def _initialize_graph(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints, )
            init = tf.global_variables_initializer()
            self.sess.run(init)

  def _load_graph(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints, )
            logger.info('Loading Model')
            ckpt = tf.train.get_checkpoint_state(self.load_model_path)
            if ckpt is None:
                logger.info('The model {0} could not be found. Make '
                            'sure you specified the right '
                            '--run-id'
                            .format(self.load_model_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
  
  
  def get_step(self,):
        return self.step
  
  def save_model(self, step):

        with self.graph.as_default():
            last_checkpoint = self.model_path + '/model-' + str(step) + '.cptk'
            self.saver.save(self.sess, last_checkpoint)
            tf.train.write_graph(self.graph, self.model_path,
                                 'raw_graph_def.pb', as_text=False)

  def _process_graph(self):

        all_nodes = [x.name for x in self.graph.as_graph_def().node]
        nodes = [x for x in all_nodes if x in self.possible_output_nodes]
        logger.info('List of nodes to export for brain :' + self.brain.brain_name)
        for n in nodes:
            logger.info('\t' + n)
        return nodes
      
     
      
  def update(self, inputs, true_label, true_label_mask, input_times):
        
        feed_dict = {
                     self.model.batch_size: len(inputs),
                     self.model.visual_in:  inputs,
                     self.model.vector_in:  input_times,
                     self.model.true_label: true_label,
                     self.model.true_label_mask: true_label_mask,
                     self.model.is_train_mode: True,}
        self.has_updated = True
        run_out = self._execute_model(feed_dict, self.update_dict)
        self.stats['train_loss'].append(run_out['loss'])
        
        return run_out
      


  def infer(self, inputs, input_times, true_label_mask):

        feed_dict = {
                     self.model.batch_size: len(inputs),
                     self.model.visual_in: inputs,
                     self.model.vector_in: input_times,
                     self.model.true_label_mask: true_label_mask,
                     self.model.is_train_mode: False}

        run_out = self._execute_model(feed_dict, self.inference_dict)

        return run_out


  def evaluate(self, inputs, true_label, true_label_mask, input_times):

        feed_dict = {
                     self.model.batch_size: len(inputs),
                     self.model.visual_in: inputs,
                     self.model.vector_in: input_times,
                     self.model.true_label: true_label,
                     self.model.true_label_mask: true_label_mask,
                     self.model.is_train_mode: False}

        run_out = self._execute_model(feed_dict, self.evaluate_dict)
        self.stats['eval_loss'].append(run_out['loss'])

        return run_out

  def _execute_model(self, feed_dict, out_dict):

        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

  def write_summary(self, global_step, lesson_num=0):

        if 1:
            is_training = "Training." if self.is_training  else "Not Training."
            
            summary = tf.Summary()
            for key in self.stats_aux:
                summary.value.add(tag='{}'.format(key), simple_value=float(self.stats_aux[key]))
            for key in self.stats:
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag='{}'.format(key), simple_value=stat_mean)
                    self.stats[key] = []
            self.summary_writer.add_summary(summary, global_step)
            self.summary_writer.flush()

  def write_tensorboard_text(self, key, input_dict):

        try:
            with tf.Session() as sess:
                s_op = tf.summary.text(key, tf.convert_to_tensor(
                    ([[str(x), str(input_dict[x])] for x in input_dict])))
                s = sess.run(s_op)
                self.summary_writer.add_summary(s, self.get_step)
        except:
            logger.info(
                "Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
            pass
          
          
          
