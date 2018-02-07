# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:32:23 2017

@author: Andrei
"""

import numpy as np
import pandas as pd
import os
import json


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

def GetDataBatch(sql_eng, car_id, predictors, predictor_names, start_date, end_date):
  """
  uses working MSSQLEngine to connect to GDCB database and retrieve batches of 
  sequential merged predictor data filling gaps in sequences with duplicated data
  each predictor will be sequentially extracted from database and finally the 
  sequences will be merged (outer join) based on timestamp index
  """
  assert type(start_date) == str
  assert type(end_date) == str
    
  m = pd.DataFrame()

  for i,predictor in enumerate(predictors):
    pred_name = predictor_names[i]
    s_sql = "SELECT TimeStamp, ViewVal as {_pred_name} FROM RawData WHERE CarID={_car_id} AND CodeID={_code_id} ".format(
        _car_id = car_id,
        _pred_name = pred_name,
        _code_id = predictor)
    s_sql+= " AND TimeStamp>='{_d_start}' AND TimeStamp<='{_d_end}' ".format(
        _d_start = start_date, 
        _d_end = end_date)
    df = sql_eng.Select(s_sql, caching = False)
    df.set_index(df.TimeStamp, inplace = True)
    df.drop("TimeStamp", axis = 1, inplace = True)
    m = pd.merge(m,df, how='outer', left_index = True, right_index = True)

  print(m)
  m.fillna(method="ffill", inplace = True)
  m.fillna(method="bfill", inplace = True)
  return m
  

if __name__=="__main__":
  entity_name = 53
  logger_module = load_module('logger','logger.py')
  logger = logger_module.Logger(lib_name = "test", 
                              config_file = "config.txt",
                              log_suffix = str(entity_name),
                              TF_KERAS = False)
  azure_helper_module = load_module('azure_helper', 'azure_helper.py')
  sql_eng = azure_helper_module.MSSQLHelper(parent_log = logger)
  predictor_names = logger.config_data["PREDICTOR_NAMES"]
  predictors = logger.config_data["PREDICTORS"]
  df1 = GetDataBatch(sql_eng, 53, predictors, predictor_names, '2017-06-23','2017-07-11')
  df2 = GetDataBatch(sql_eng, 53, predictors, predictor_names, '2017-07-11','2017-11-14')