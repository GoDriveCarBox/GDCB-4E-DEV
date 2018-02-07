# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 04:58:32 2017

@author: Damian

@description: Persistent Anomaly Detector Class

@modifid:
  2017-10-10 Created
  2017-11-12 v0.1 ready - no object persistance - 
  2017-11-13 added testing
"""
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime as dt
from time import sleep

__VERSION__   = "0.2.1"
__NAME__      = "Persistent Anomaly Detector"
__SNAME__     = "PAD"

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


class PersistentAnomalyDetector: 
  """
  Persistent Anomaly Detector
  """
  def __init__(self, config_file = 'config.txt', threshold = 0.02,
               entity_name = -1, model_name = "PAD"):
    """
    Constructor - initialize all props of Persistent Anomaly Detector
    """
    self.__version__ = __VERSION__
    self.__name__ = __NAME__
    self.__sname__ = __SNAME__
    self.CONFIG = None

    logger_module = load_module('logger','logger.py')
    self.logger = logger_module.Logger(lib_name = "PADv0", 
                                config_file = config_file,
                                log_suffix = str(entity_name),
                                TF_KERAS = False,
                                HTML = True)

    azure_helper_module = load_module('azure_helper', 'azure_helper.py')
    self.sql_eng = azure_helper_module.MSSQLHelper(parent_log = self.logger)

    self.features = dict()
    self.overall = dict()
    self.fitted = False
    self.threshold = threshold
    self._load_config(config_file)
    self.entity_name = entity_name
    self.model_name = model_name
    
    return
  
  
  def _prop_gdens(self, std_sq, mean, val):
    """
    Gaussian density function for one feature
    INEFFICIENT IMPLEMENTATION
    """
    res_sq = (val - mean)**2
    _sqrt = np.sqrt(2*np.pi*std_sq)
    _exp = np.exp(-res_sq / (2*std_sq))
    _result = 1 / _sqrt  * _exp
    return _result
  
  def _obs_gdens(self,obs):
    """
    multi-variate Gaussian density function
    INEFFICIENT IMPLEMENTATION
    """
    _result = 1
    for key in self.features.keys():
      val = obs[key]
      std_sq = self.features[key]['std_sq']
      mean = self.features[key]['mean']
      prop_gdens = self._prop_gdens(std_sq, mean, val) 
      _result *= prop_gdens
    return _result
  
  def predict_proba(self, df_X):
    """
    predict dataframe observation by observation
    returns percentage of max observation Gaussian density function
    """
    _result = df_X.apply(self._obs_gdens, axis = 1).values
    _result /= self.overall['overall_max_gdf']
    return _result
  
  def predict(self, df_X):
    """
    predict dataframe observation by observation
    returns 1 for valid and 0 for anomaly
    """
    _result = self.predict_proba(df_X) >= self.threshold
    return _result
  
  def predict_anomaly_prob(self, df_X):
    """
    predicts anomalies with probabilities
    0.5 means anomaly at the threshold; ~1 means certainly anomaly
    """
    _result = self.predict_proba(df_X)
    _result = 0.5 - ((_result - self.threshold) / (2 * self.threshold))
    
    return _result
  
  def predict_anomaly(self, df_X):
    """
    predict dataframe observation by observation
    returns 1 for valid and 0 for anomaly
    """
    _result = self.predict_anomaly_prob(df_X) >= 0.5
    return _result
  
  def _load_config(self, config_file = 'config.txt'):
    """
    loads configuration JSON txt file
    """
    self.CONFIG = self.logger.config_data

    return
  
  def fit(self, df_X_train, excluded_fields=None):
    """
    fit model with full batch initial data
    it will accept batches and continue if allready fitted
    """
    accepted_fields = list(df_X_train.columns)
    if excluded_fields != None:
      accepted_fields = list(set(accepted_fields)-set(excluded_fields))
    for field in accepted_fields:
      if not (field in self.features.keys()):
        self.features[field] = {'count':0, 'mean':0, 'std_sq':0} # init feature
      old_mean = self.features[field]['mean']
      old_std_sq = self.features[field]['std_sq']
      old_count = self.features[field]['count']
      new_mean = df_X_train[field].mean()
      new_count = df_X_train.shape[0]
      new_std_sq = df_X_train[field].values.std()**2
      
      old_sum = old_mean * old_count
      new_sum = new_mean * new_count

      final_count = old_count + new_count
      final_mean = (old_sum + new_sum) / final_count
      
      v1 = old_count * (old_std_sq + old_mean**2)
      v2 = new_count * (new_std_sq + new_mean**2)
      
      v3 = (v1 + v2) / final_count
      std_sq = v3  - final_mean**2

      final_std_sq = std_sq

      self.features[field]['mean'] = final_mean
      self.features[field]['count'] = final_count
      self.features[field]['std_sq'] = final_std_sq
    
    self.overall['overall_max_gdf'] = 1
    for field in accepted_fields:
      self.overall['overall_max_gdf'] *= 1/np.sqrt(2*np.pi*self.features[field]['std_sq'])
    self.fitted = True
    self._save_model()
      
    return

  def fit_predict(self, df_X):
    """
    fist df_X dataframe and returns inference vector for all obs
    """
    self.fit(df_X)
    return self.predict(df_X)

  def fit_predict_proba(self, df_X):
    self.fit(df_X)
    return self.predict_proba(df_X)
  
  def fit_predict_anomaly_prob(self, df_X):
    self.fit(df_X)
    return self.predict_anomaly_prob(df_X)
  
  def fit_predict_anomaly(self, df_X):
    self.fit(df_X)
    return self.predict_anomaly(df_X)
  
  def _save_model(self):
    """
    saves model status to server
    """
    # save self.features
    s_sql = ''
    for key in self.features.keys():
      for param in self.features[key].keys():
        val = self.features[key][param]
        s_sql+=" begin tran"
        s_sql+="  if exists (select * from dbo.Models with (updlock,serializable) where CarID={} AND Model='{}' AND Categ ='FEATURES' AND Field='{}' AND Param ='{}')".format(
            self.entity_name,
            self.model_name,
            key,
            param
          )
        s_sql+="      begin"
        s_sql+="         update dbo.Models set Value = {}".format(val)
        s_sql+="         where CarID={} AND Model='{}' AND Categ='FEATURES' AND Field='{}' AND Param='{}' ".format(self.entity_name, self.model_name, key, param)
        s_sql+="      end"
        s_sql+="  else"
        s_sql+="     begin"
        s_sql+="         insert into dbo.Models ([CarID], [Model],[Categ],[Field],[Param],[Value])"
        s_sql+="         values ({},'{}','FEATURES','{}','{}',{})".format(self.entity_name, self.model_name, key, param, val)
        s_sql+="     end"
        s_sql+=" commit tran "

    # save self.overall
    for param in self.overall.keys():
        val = self.overall[param]
        s_sql+=" begin tran"
        s_sql+="  if exists (select * from dbo.Models with (updlock,serializable) where CarID={} AND Model='{}' AND Categ ='OVERALL' AND Field='' AND Param ='{}')".format(
            self.entity_name,
            self.model_name,
            param
          )
        s_sql+="      begin"
        s_sql+="         update dbo.Models set Value = {}".format(val)
        s_sql+="         where CarID={} AND Model='{}' AND Categ='OVERALL' AND Field='' AND Param='{}' ".format(self.entity_name, self.model_name, param)
        s_sql+="      end"
        s_sql+="  else"
        s_sql+="     begin"
        s_sql+="         insert into dbo.Models ([CarID], [Model],[Categ],[Field],[Param],[Value])"
        s_sql+="         values ({},'{}','OVERALL','','{}',{})".format(self.entity_name, self.model_name, param, val)
        s_sql+="     end"
        s_sql+=" commit tran "

    self.logger.VerboseLog('Saved features and summary from CarID={}'.format(self.entity_name))
    self.sql_eng.ExecInsert(s_sql)
    return
  
  def _load_model(self):
    """
    loads model status from server
    """
    
    # load self.features
    
    # load self.overall
    return

  def GetDataBatch(self, start_date, end_date):
    """
    uses working MSSQLEngine to connect to GDCB database and retrieve batches of 
    sequential merged predictor data filling gaps in sequences with duplicated data
    each predictor will be sequentially extracted from database and finally the 
    sequences will be merged (outer join) based on timestamp index
    """
    assert type(start_date) == str
    assert type(end_date) == str

    m = pd.DataFrame()
    
    predictor_names = self.CONFIG["PREDICTOR_NAMES"]
    predictors = self.CONFIG["PREDICTORS"]
    self.logger.VerboseLog('Fetching batch from the database with data between {} and {}'.format(start_date, end_date))

    for i,predictor in enumerate(predictors):
      pred_name = predictor_names[i]
      s_sql = "SELECT TimeStamp, ViewVal as {_pred_name} FROM RawData WHERE CarID={_car_id} AND CodeID={_code_id} ".format(
          _car_id = self.entity_name,
          _pred_name = pred_name,
          _code_id = predictor)
      s_sql+= " AND TimeStamp>='{_d_start}' AND TimeStamp<'{_d_end}' ".format(
          _d_start = start_date, 
          _d_end = end_date)
      df = self.sql_eng.Select(s_sql, caching = False)
      df.set_index(df.TimeStamp, inplace = True)
      df.drop("TimeStamp", axis = 1, inplace = True)
      m = pd.merge(m,df, how='outer', left_index = True, right_index = True)

    m.fillna(method="ffill", inplace = True)
    m.fillna(method="bfill", inplace = True)
    self.logger.VerboseLog('Finished creating the batch!')
    return m


if __name__=="__main__":
  
  pd.options.display.float_format = '{:.4f}'.format
  clf = PersistentAnomalyDetector(entity_name = 53)
  
  s_batch_start = '2017-06-01'
  s_batch_end = '2017-11-14'
  df_train = clf.GetDataBatch(s_batch_start, s_batch_end)
  s_batch_start = s_batch_end
  clf.fit(df_train)
  
  
  while True:
    clf.logger.VerboseLog('Waiting 30s in order to generate new data in the database...')
    sleep(30)

    batch_end_tstamp = dt.now()
    s_batch_end = batch_end_tstamp.strftime('%Y-%m-%d %H:%M:%S')    
    df_test = clf.GetDataBatch(s_batch_start, s_batch_end)
    s_batch_start = s_batch_end
    
  
    if df_test.shape[0] > 0:
      df_test['Anomaly_Prob'] = clf.predict_anomaly_prob(df_test) * 100
      df_test['Good'] = clf.fit_predict(df_test)
      
      df_anomaly = df_test[df_test['Good'] == False]
      print(df_anomaly)
      for index, row in df_anomaly.iterrows():
        clf.logger.VerboseLog("{_anomaly_prob}% DETECTED AN ANOMALY for the car with ID = {_car_id}."\
                              "The TimeStamp of the anomaly is {_timestamp}".format(
                                  _anomaly_prob = row.Anomaly_Prob
                                  _car_id = clf.entity_name,
                                  _timestamp = index))
  
  
  '''
  df_train = pd.DataFrame()
  np.random.seed(seed=1234)
  for batch in range(5):
    max_int = 10*batch+10
    np_batch = np.random.randint(1,max_int,size=(5,3))
    df_batch = pd.DataFrame(np_batch)
    df_batch.columns = ['A','B','C'] 
    clf.fit(df_batch)
    df_train = pd.concat([df_train,df_batch])

  np_test = np.random.randint(1,max_int+20,size=(10,3))
  df_test = pd.DataFrame(np_test)
  df_test.columns = ['A','B','C'] 

  for f in df_train.columns:
    print("{}: {}\n  {}\n\n".format(f,clf.features[f], df_train[f].values.std()**2))

  df_train['Prob'] = clf.predict_proba(df_train) * 100
  df_train["Good"] = clf.predict(df_train)
  print("\nTrain data:")
  print(df_train)

  df_test['Prob'] = clf.predict_proba(df_test) * 100
  df_test["Good"] = clf.predict(df_test)
  print("\nTest data:")
  print(df_test)
  
  df_test_fit_predict = pd.DataFrame(np_test)
  df_test_fit_predict.columns = ['A','B','C'] 
  df_test_fit_predict['Prob'] = clf.fit_predict_proba(df_test_fit_predict) * 100
  df_test_fit_predict['Good'] = clf.predict(df_test_fit_predict)
  print("\nFit_predict (add test data to training):")
  print(df_test_fit_predict)
  '''