# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:36:30 2018

@author: Andrei Ionut Damian

@modified:
  2018-04-12 added debug info
  2018-04-13 added support for database config with master-detail
  2018-04-17 finished support for multiple PADs
  

TODO:
  - implement threading

"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
from time import sleep

import os
from PAD import PersistentAnomalyDetector


def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    raise Exception("Couldn't find google drive folder!")

  utils_path = os.path.join(drive_path, "_pyutils")
  print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
  module_lib   = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
  print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return module_lib



class PADManager:
  def __init__(self, config_file='config.txt', DEBUG=True):
    pd.set_option('display.width', 500)
    pd.options.display.float_format = '{:.6f}'.format
    np.set_printoptions(suppress=True, precision=7, floatmode='fixed')    
    
    logger_module = load_module('logger','logger.py')

    self.logger = logger_module.Logger(lib_name = "PADmgr", 
                                config_file = config_file,
                                TF_KERAS = False,
                                HTML = True)
    

    azure_helper_module = load_module('azure_helper', 'azure_helper.py')
    self.sql_eng = azure_helper_module.MSSQLHelper(parent_log = self.logger)    
    self.clfs = []
    self.DEBUG = DEBUG
    self._get_entities_config()
    self.events_df = self.sql_eng.GetEmptyTable("PADEvents")
    
    return
  
  
  def _get_entities_config(self):    
    s_sql = 'select * from CarPredictors'
    self.df_cars = self.sql_eng.Select(s_sql, caching=False)
    self.cars = {}
    for idx in self.df_cars.index:
      car_id = self.df_cars.loc[idx,'CarID']
      train_start = str(self.df_cars.loc[idx,'TrainStart'])
      train_end = str(self.df_cars.loc[idx,'TrainEnd'])
      cfg = {'TrainStart': train_start, 'TrainEnd':train_end}
      self.logger.P("Loading predictors config for entity: {}".format(car_id))
      s_sql_car = 'select * from CarPredictorsDetail where CarID={}'.format(car_id)
      df_car = self.sql_eng.Select(s_sql_car, caching=False)
      cfg['PREDICTORS'] = list(df_car['CodeID'])
      cfg['PREDICTOR_NAMES'] = list(df_car['Name'].str.rstrip())
      self.cars[car_id] = cfg
    return
  
  
  def add(self,carid, cfg_dict):
    clf = PersistentAnomalyDetector(entity_name = carid, sql_eng=self.sql_eng,
                                    logger=self.logger)
    clf.set_config(cfg_dict)
    clf._load_model()      
    self.clfs.append(clf)
    return
  
  def run(self, train=True, run_from_date=''):
    assert run_from_date != '', "run_from_date must be 'yyyy-mm-dd'"
    for car_id in self.cars.keys():
      car_cfg = self.cars[car_id]
      self.add(carid=car_id, cfg_dict=car_cfg)
      
    self.summary()
    
    if train:
      for clf in self.clfs:
        if not clf.fitted:
          clf.InitFit()
            
    # run predict on all entities
    s_batch_start = run_from_date
    for clf in self.clfs:
      if str(clf.entity_name) != "57":
        continue
      
      batch_end_tstamp = dt.now()
      s_batch_end = batch_end_tstamp.strftime('%Y-%m-%d %H:%M:%S')    
      self._predict_iter(clf, s_batch_start, s_batch_end)        
    return
  
  
  def _predict_iter(self, clf, s_batch_start, s_batch_end):
      #clf.logger.VerboseLog('Waiting {}s in order to generate new data in the database...'.format(wait_time))
      #sleep(wait_time)
  
      df_test = clf.GetDataBatch(s_batch_start, s_batch_end)
      res = None
      
      #print(s_batch_start, s_batch_end)
      #print(df_test.head())
    
      if df_test.shape[0] > 0:
        df_test['Anomaly_Prob'] = clf.predict_anomaly_prob(df_test) * 100
        df_test['GProba'] = clf.predict_proba(df_test) * 100
        
        df_test['Good'] = clf.predict_anomaly(df_test)      
        df_anomaly = df_test[df_test['Good'] == False]
        
        for index, row in df_anomaly.iterrows():
          clf.analize_obs(index, row, self.cars[clf.entity_name], 
                          save_anomaly = True)
          break
          
        '''
        for i in range(df_anomaly.shape[0]):
          clf.analize_obs(df_anomaly.iloc[i], self.cars[clf.entity_name], 
                          save_anomaly = True)
          break
        '''
          
        if not self.DEBUG:
          for index, row in df_anomaly.iterrows():
            clf.logger.VerboseLog("{_anomaly_prob:.1f}% DETECTED AN ANOMALY for the car with ID = {_car_id}."\
                                  "Anomaly timestamp: {_timestamp::%Y-%m-%d %H:%M:%S}".format(
                                      _anomaly_prob = row.Anomaly_Prob,
                                      _car_id = clf.entity_name,
                                      _timestamp = index))
          #endfor
        #endif
      #endif
      return res
    
  
  def summary(self):
    for c in self.clfs:
      _n = c.name
      _id = c.entity_name
      _p = c.PREDS
      _pn = c.PRED_NAMES
      _ts = c.TRAIN_START
      _te = c.TRAIN_END
      self.logger.P("="*80)
      self.logger.P("{}: {} with {} predictors {} trained on '{}'-'{}'".format(
          _n, _id, len(_p), _pn, _ts, _te))
      self.logger.P("="*80)
      
    return
  
  def P(self,_str, show_time):
    self.logger.P(_str, show_time=show_time)
    return
    
  
  
  
if __name__ == '__main__':
  pd.set_option('display.width', 500)
  pd.options.display.float_format = '{:.6f}'.format
  np.set_printoptions(suppress=True, precision=7, floatmode='fixed')
  
  app = PADManager()
  app.run(train=True, run_from_date='2018-04-05')
  

  