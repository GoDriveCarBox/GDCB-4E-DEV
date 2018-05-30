# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 04:58:32 2017

@author: Damian

@description: Persistent Anomaly Detector Class

@modifid:
  2017-10-10 Created
  2017-11-12 v0.1 ready - no object persistance - 
  2017-11-13 added testing
  2018-04-04 debugging
  2018-04-12 added support for config dict
  2018-04-13 added support for database config
  
  
TODO:
  - testare pe date car 57  
  
  
"""
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime as dt
from time import sleep

import pprint
from PAD_saver import AnomalyPersistor

from dateutil.parser import parse

def is_date(_str):
  try: 
      parse(_str)
      return True
  except ValueError:
      return False 

#from PAD_saver import AnomalyPersistor

__VERSION__   = "1.1.3"
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
               entity_name = -1, model_name = "PAD",
               ddof = 0, sql_eng=None, logger=None):
    """
    Constructor - initialize all props of Persistent Anomaly Detector
    """
    self.entity_name = entity_name
    self._ddof = ddof
    self.__version__ = __VERSION__
    self.__name__ = __NAME__
    self.__sname__ = __SNAME__
    self.CONFIG = None
    
    
    if logger is None:
      logger_module = load_module('logger','logger.py')
      self.logger = logger_module.Logger(lib_name = "PADv0", 
                                  config_file = config_file,
                                  log_suffix = str(entity_name),
                                  TF_KERAS = False,
                                  HTML = True)
    else:
      self.logger = logger
    
    self.P("Initializing PersistentAnomalyDetector v.{} for EntityID: {}".format(
        self.__version__, self.entity_name))

    if sql_eng is None:
      azure_helper_module = load_module('azure_helper', 'azure_helper.py')
      self.sql_eng = azure_helper_module.MSSQLHelper(parent_log = self.logger)
    else:
      self.sql_eng = sql_eng

    self.features = dict()
    self.overall = dict()
    self.fitted = False
    self.threshold = threshold
    self._load_config()
    self.model_name = model_name
    self.name = "{}_{}".format(self.model_name,self.entity_name)
    
    return
  
  
  def _gdens(self, std_sq, mean, val):
    """
    Gaussian density function for one feature
    INEFFICIENT IMPLEMENTATION
    """
    res_sq = (val - mean)**2
    _sqrt = np.sqrt(2 * np.pi * std_sq)
    
    '''
    if std_sq == 0.0:
      std_sq = 0.001
    '''
    
    
    _exp = np.exp(-res_sq / (2*std_sq))
    _result = (1 / _sqrt) * _exp
    return _result
  
  def _obs_gdens(self, obs):
    """
    multi-variate Gaussian density function
    INEFFICIENT IMPLEMENTATION
    """
    _result = 1
    for key in self.features.keys():
      
      '''
      if not key in obs:
        continue
      '''
      
      val = obs[key]
      std_sq = self.features[key]['std_sq']
      mean = self.features[key]['mean']
      _gdens = self._gdens(std_sq, mean, val)  
      _gprob = _gdens / self.features[key]['max_gdf']
      if _gprob < self.threshold:
        _result *= _gprob
    return _result
  
  def predict_proba(self, df_X):
    """
    predict dataframe observation by observation
    returns percentage of max observation Gaussian density function
    """
    _result = df_X.apply(self._obs_gdens, axis = 1).values
    #_result /= self.overall['overall_max_gdf']
    return _result
  
  def _analize_data(self, obs):
    """
    analyze a particular observation
    """
    _result = {}
    _v = 1
    _g = 1
    probas=[]
    for key in self.features.keys():
      
      '''
      if not key in obs:
        continue
      '''
      
      _prop = {}
      val = obs[key]
      std_sq = self.features[key]['std_sq']
      mean = self.features[key]['mean']
      _gdens = self._gdens(std_sq, mean, val)           
      proba = _gdens / self.features[key]['max_gdf']
      _prop['VALUE'] = val
      _prop['PROBA'] = proba
      _prop['GDENS'] = _gdens
      _prop['FAULT'] = proba < self.threshold
      aproba = self._gproba_to_aproba(proba)
      _prop['FAULT_PROBA'] = aproba
      if aproba >= 0.5:
        probas.append(aproba)
      _result[key] = _prop
      _v *= proba
      _g *= _gdens
    
    
    _result['TOTAL_GPROBAP'] = _v
    _result['TOTAL_GPROBAB'] = _g / self.overall['overall_max_gdf']
    _result['TOTAL_GDENS'] = _g
    _result['TOTAL_APROBA'] = np.max(probas)+0.01
    
    return _result

  def save_anomaly_event(self, code, proba):
    """
    this method saves the event based on key (CodeID) and aproba (Proba)
    in the PADEvents table using sql_eng object and constructed 
    SQL INSERT STATEMENT
    """
    tstmp = dt.now()
    s_time = tstmp.strftime('%Y-%m-%d %H:%M:%S')    
    s_car = '{}'.format(self.entity_name)
    s_code = '{}'.format(code)
    s_prob = '{:.4f}'.format(proba)
    sql = "INSERT INTO dbo.PADEvents ([EventTime], [CarID], [CodeID], [Proba]) VALUES "
    sql += "('"+s_time+"',"+s_car+","+s_code+","+s_prob+")"
    self.P('Saving car {} event with {}% for {}'.format(
        s_car, s_prob, s_code))
    _res = self.sql_eng.ExecInsert(sql)
    if _res:
      self.P('Done saving event.')
    else:
      self.P('Event saving FAILED!')
    return
    

  def analize_obs(self, obs, save_anomaly=True):
    #weight = 0.5
    self._reset_faults()    
    res = self._analize_data(obs)
    self._add_fault(res)
    aproba = res['TOTAL_APROBA']
    if  aproba >= 0.5:
      if aproba > 1.0:
        aproba = 1.0
      self.P("Anomaly detected with {:.1f}% probability:".format(aproba*100))
      for key in self.features.keys():
        
        '''
        if not key in obs:
          continue
        '''
         
        if res[key]['FAULT']:
          val = res[key]['VALUE']
          gproba = res[key]['PROBA']
          aproba = self._gproba_to_aproba(gproba)
          self.P("  {:10s} has FAULT {:.2f}% with value {:.2f}".format(
              key, aproba*100, val))
          if save_anomaly:
            self.save_anomaly_event(code=key, proba=aproba)
      _faults, _text = self._get_faults()
      self.P("  PREDICTED FUTURE FAILURES: {}".format(_text))
    return res, val

    
  
  def _reset_faults(self):
    for key in self.fault_list.keys():
      self.fault_list[key]['SCORE'] = 0
    return
  
  
  def _add_fault(self,adict):
    for key in self.fault_list.keys():
      for akey in adict.keys():
        if akey in self.fault_list[key]['DETECTORS'].keys():
          aproba = adict[akey]['FAULT_PROBA']
          self.fault_list[key]['SCORE'] += self.fault_list[key]['DETECTORS'][akey] * aproba          
    return
  
  
  def _get_faults(self):
    lst = []
    _result = []
    _text = ""
    vtotal = 0.00001
    for key in self.fault_list.keys():
      _score = self.fault_list[key]['SCORE']
      vtotal += _score
      _fault = self.fault_list[key]['NAME']
      lst.append([_fault,_score])
    lst = sorted(lst, key=lambda tup: tup[1], reverse=True)
    for i in range(len(lst)):
      lst[i][1] /=  vtotal
      if lst[i][1] > 0.4:
        _result.append(lst[i])
        _text += "{}: {:.2f}%".format(lst[i][0],lst[i][1]*100)
    return _result, _text
    
  
  def _set_vars(self):
    self.PREDS = self.CONFIG['PREDICTORS']
    self.PRED_NAMES = self.CONFIG['PREDICTOR_NAMES']
    self.TRAIN_START = self.CONFIG['TrainStart']
    self.TRAIN_END = self.CONFIG['TrainEnd']
    self.PREDICTORS = dict(zip(self.PREDS, self.PRED_NAMES ))
    return
  
  
  def predict(self, df_X):
    """
    predict dataframe observation by observation
    returns 1 for valid and 0 for anomaly
    """
    _result = self.predict_proba(df_X) >= self.threshold
    return _result
  
  def _gproba_to_aproba(self, val, weight=1):
    _res1 =  0.5 - ((val - self.threshold*weight) / (2 * self.threshold*weight))
    if type(_res1) is np.ndarray:
      _res = np.where(0>_res1,0,_res1)
    else:
      _res = max(0, _res1)
    return _res
  
  def predict_anomaly_prob(self, df_X):
    """
    predicts anomalies with probabilities
    0.5 means anomaly at the threshold; ~1 means certainly anomaly
    """
    _result1 = self.predict_proba(df_X)
    _result = self._gproba_to_aproba(_result1)
    return _result
  
  def predict_anomaly(self, df_X):
    """
    predict dataframe observation by observation
    returns 1 for valid and 0 for anomaly
    """
    _result = self.predict_proba(df_X) >= self.threshold
    return _result
  

  def set_config(self, cfg_dict):
    for k in cfg_dict.keys():
      self.CONFIG[k] = cfg_dict[k]
    self._set_vars()
    return
  
  
  def _load_config(self):
    self.CONFIG = self.logger.config_data.copy()
    self.eps = self.CONFIG['EPSILON']
    self.fault_list = self.CONFIG['PREDICTED_FAULTS']
    return
  
  def _load_config_old(self):
    """
    loads configuration JSON txt file
    """
    self.CONFIG = self.logger.config_data.copy()
    
    if self.CONFIG['PREDICTORS'] == "0":
      select_query = "SELECT ID, Description FROM Codes"
      codes_df = self.sql_eng.Select(select_query)
       
      self.CONFIG['PREDICTORS'] = list(codes_df['ID'].values)
      self.CONFIG['PREDICTORS'] = [int(e) for e in self.CONFIG['PREDICTORS']]
      self.CONFIG['PREDICTOR_NAMES'] = list(codes_df['Description'].values)
      self.CONFIG['PREDICTOR_NAMES'] = ['_'.join(e.split()) for e in self.CONFIG['PREDICTOR_NAMES']]
      print(self.CONFIG['PREDICTOR_NAMES'])
      
      
    with open("full_config.txt", 'w') as fp:
      json.dump(self.CONFIG, fp, indent = 4)
      
    self.fault_list = self.CONFIG['PREDICTED_FAULTS']
    self._set_vars()
    
    return
  
  def fit(self, df_X_train, excluded_fields=None):
    """
    fit model with full batch initial data
    it will accept batches and continue if allready fitted
    """
    self.P("PAD #{} fitting {} obs".format(self.entity_name, df_X_train.shape[0]))
    if df_X_train.shape[0] == 0:
      self.logger.VerboseLog("Fit received empty dataframe")
      return
    accepted_fields = list(df_X_train.columns)
    if excluded_fields != None:
      accepted_fields = list(set(accepted_fields)-set(excluded_fields))
    for field in accepted_fields:
      if not (field in self.features.keys()):
        self.features[field] = {'count':0, 'mean':0, 'std_sq':0, 'min':0, 'max':0, 
                                'std':0, 'max_gdf':0} # init feature
      old_mean = self.features[field]['mean']
      old_std_sq = self.features[field]['std_sq']
      old_count = self.features[field]['count']
      old_min = self.features[field]['min']
      old_max = self.features[field]['max']
      new_mean = df_X_train[field].mean()
      new_count = df_X_train.shape[0]
      if new_count == 0:
        continue
      new_std_sq = df_X_train[field].std(ddof=self._ddof)**2
      new_max = df_X_train[field].max()
      new_min = df_X_train[field].min()
      
      old_sum = old_mean * old_count
      new_sum = new_mean * new_count

      final_count = old_count + new_count
      final_mean = (old_sum + new_sum) / final_count

      ### MUST BE MODIFIEF FOR DDOF == 1      
      v1 = old_count * (old_std_sq + old_mean**2)
      v2 = new_count * (new_std_sq + new_mean**2)
      
      v3 = (v1 + v2) / (final_count)
      std_sq = v3  - final_mean**2
      ###

      final_std_sq = std_sq

      self.features[field]['min'] = min(old_min, new_min)
      self.features[field]['max'] = max(old_max, new_max)      
      self.features[field]['mean'] = final_mean
      self.features[field]['count'] = final_count
      self.features[field]['std_sq'] = final_std_sq
      self.features[field]['std'] = np.sqrt(final_std_sq)
      if self.features[field]['std_sq'] != 0:
        self.features[field]['max_gdf'] = 1/np.sqrt(2*np.pi*self.features[field]['std_sq'])
      
    
    self.overall['overall_max_gdf'] = 1
    for field in accepted_fields:
      if self.features[field]['std_sq'] != 0:
        self.overall['overall_max_gdf'] *= 1/np.sqrt(2*np.pi*self.features[field]['std_sq'])
    self.fitted = True
    self.P("Done fitting {} obs".format(self.entity_name, df_X_train.shape[0]), True)
    self._save_model()
      
    return
    
  
  def P(self, _str, _show=False):
    self.logger.VerboseLog(_str, show_time=_show)
    return
  
  def _sanity_check(self, df_t):
    self.P("Performing sanity check...")
    errors = 0
    ss = df_t.std(ddof=self._ddof)
    for n in ss.index:
      v1 = round(self.features[n]['std'],7)
      v2 = round(ss[n],7)
      if v1 != v2:
        self.P("Sanity check error STD {} {:.5f} != {:.5f}".format(n, v1, v2))
        errors += 1
    ss = df_t.mean()
    for n in ss.index:
      v1 = round(self.features[n]['mean'],7)
      v2 = round(ss[n],7)
      if v1 != v2:
        self.P("Sanity check error MEAN {} {:.5f} != {:.5f}".format(n, v1, v2))
        errors += 1    
    if errors > 0:
      self.P("Done performing sanity check with ERRORS: {}".format(errors))
    else:
      self.P("Done performing sanity check with no errors.")
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
    saver = AnomalyPersistor()
    saver._save_model(self)
    
    return
  
  def _load_model(self):
    """
    loads model status from server
    """
    
    loader = AnomalyPersistor()
    if loader._load_model(self):
      self.fitted = True
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
    
    predictor_names = self.PRED_NAMES
    predictors = self.PREDS
    self.logger.VerboseLog('PAD #{} Fetching data batch between {} and {}'.format(
        self.entity_name, start_date, end_date))

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
      
      if (df is None) or (df.shape[0] == 0):
        continue
      
      ######################################
      df["TimeStamp"] = df["TimeStamp"].apply(lambda x: x.replace(microsecond=0))
      df.drop_duplicates(subset='TimeStamp', keep = 'last', inplace = True)
      ######################################
      
      df.set_index(df.TimeStamp, inplace = True)
      df.drop("TimeStamp", axis = 1, inplace = True)
      m = pd.merge(m,df, how='outer', left_index = True, right_index = True)

    m.fillna(method="ffill", inplace = True)
    m.fillna(method="bfill", inplace = True)
    self.logger.VerboseLog('Finished creating the batch!')
    return m
  
  def InitFit(self):
    assert is_date(self.TRAIN_START) and is_date(self.TRAIN_END), "Unknown dates [{}] / [{}]".format(
        clf.TRAIN_START, clf.TRAIN_END)
    self.P("Initialization Fit for PAD {}".format(self.name))
    df_train = self.GetDataBatch(self.TRAIN_START, self.TRAIN_END)
    self.fit(df_train)    
    return


if __name__=="__main__":
  pd.set_option('display.width', 500)
  pd.options.display.float_format = '{:.6f}'.format
  np.set_printoptions(suppress=True, precision=7, floatmode='fixed')
  
  clf = PersistentAnomalyDetector(entity_name = 57)
  
  s_batch_start1 = '2018-01-28'
  s_batch_end1 = '2018-02-05'
  s_batch_start2 = s_batch_end1
  s_batch_end2 = '2018-02-28'
  #nr_obs = 105

  df_full = clf.GetDataBatch(s_batch_start1, s_batch_end2)
  
  #df_train1 = df_full.iloc[:nr_obs,:].copy() #clf.GetDataBatch(s_batch_start1, s_batch_end1)
  #df_train2 = df_full.iloc[nr_obs:,:].copy() #clf.GetDataBatch(s_batch_start2, s_batch_end2)

  #clf.fit(df_train1)

  #clf.fit(df_train2)
  
  clf.fit(df_full)
  
  clf._sanity_check(df_full)
  
  wait_time = 5
  s_batch_start = s_batch_end2
  
  DEBUG = True
  
  last_values = None
  while True:
    clf.logger.VerboseLog('Waiting {}s in order to generate new data in the database...'.format(wait_time))
    sleep(wait_time)

    batch_end_tstamp = dt.now()
    s_batch_end = batch_end_tstamp.strftime('%Y-%m-%d %H:%M:%S')    
    df_test = clf.GetDataBatch(s_batch_start, s_batch_end)
    s_batch_start = s_batch_end
    
  
    if df_test.shape[0] > 0:
      df_test['Anomaly_Prob'] = clf.predict_anomaly_prob(df_test) * 100
      df_test['GProba'] = clf.predict_proba(df_test) * 100
      
      df_test['Good'] = clf.predict_anomaly(df_test)      
      df_anomaly = df_test[df_test['Good'] == False]
      if DEBUG:
        i = 0
        #for i in range(df_anomaly.shape[0]):
        res, val = clf.analize_obs(df_anomaly.iloc[i])
        clf.logger.VerboseLog(df_test.head())
      
      if not DEBUG:
        for index, row in df_anomaly.iterrows():
          clf.logger.VerboseLog("{_anomaly_prob:.1f}% DETECTED AN ANOMALY for the car with ID = {_car_id}."\
                                "Anomaly timestamp: {_timestamp::%Y-%m-%d %H:%M:%S}".format(
                                    _anomaly_prob = row.Anomaly_Prob,
                                    _car_id = clf.entity_name,
                                    _timestamp = index))
      else:
        break
  
  
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