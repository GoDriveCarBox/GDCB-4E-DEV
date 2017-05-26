# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:45:41 2017

@author: Andrei
"""


__author__     = "Andrei Ionut DAMIAN"
__project__    = "GoDriveCarBox"
__copyright__  = "Copyright 2007 4E Software"
__credits__    = ["Andrei Simion"]
__license__    = "GPL"
__version__    = "0.1.1"
__maintainer__ = "Andrei Ionut DAMIAN"
__email__      = "damian@4esoft.ro"
__status__     = "Production"
__library__    = "DATA EXPLORER"
__created__    = "2017-01-25"
__modified__   = "2017-05-25"
__lib__        = "GDCBDE"


from gdcb_azure_helper import MSSQLHelper
import pandas as pd
from datetime import datetime as dt
import numpy as np
import os
import json

class GDCBExplorer:
  """
  GDCB Data Explorer main class
   - uploads data to Azure via GDCB Azure Helper engine
   - downloads data for model training and prediction
   - acts as a general data broker
  """
  def __init__(self):
    self.FULL_DEBUG = True
    pd.options.display.float_format = '{:,.3f}'.format
    pd.set_option('expand_frame_repr', False)
    np.set_printoptions(precision = 3, suppress = True)


    self.MODULE = "{} v{}".format(__library__,__version__)
    self.s_prefix = dt.strftime(dt.now(),'%Y%m%d')
    self.s_prefix+= "_"
    self.s_prefix+=dt.strftime(dt.now(),'%H%M')
    self.s_prefix+= "_"
    self.cwd = os.getcwd()
    self.save_folder = os.path.join(self.cwd,"temp")
    self.log_file = os.path.join(self.save_folder,        
                                 self.s_prefix + __lib__+"_log.txt")
    nowtime = dt.now()
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(__lib__))
    print(strnowtime+"Init log: {}".format(self.log_file))
    
    if not os.path.exists(self.save_folder):
        print(strnowtime+"CREATED TEMP LOG FOLDER: {}".format(self.save_folder))
        os.makedirs(self.save_folder)
    else:
        print(strnowtime+"TEMP LOG FOLDER: {}".format(self.save_folder))
    self.sql_eng = MSSQLHelper(parent_log = self)
    self.setup_folder()
    self._logger("Work folder: [{}]".format(self.save_folder))


    self._logger("INIT "+self.MODULE)

    if self.FULL_DEBUG:
        self._logger(self.s_prefix)
        self._logger("__name__: {}".format(__name__))
        self._logger("__file__: {}".format(__file__))
    self._load_config()
    return
  
  def _logger(self, logstr, show = True):
    """ 
    log processing method 
    """
    if not hasattr(self, 'log'):        
        self.log = list()
    nowtime = dt.now()
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(__lib__))
    logstr = strnowtime + logstr
    self.log.append(logstr)
    if show:
        print(logstr, flush = True)
    try:
        log_output = open(self.log_file, 'w')
        for log_item in self.log:
          log_output.write("%s\n" % log_item)
        log_output.close()
    except:
        print(strnowtime+"Log write error !", flush = True)
    return
  
  
  def setup_folder(self):
    """
    Setup folders for app
    """
    self.s_prefix = dt.strftime(dt.now(),'%Y%m%d')
    self.s_prefix+= "_"
    self.s_prefix+=dt.strftime(dt.now(),'%H%M')
    self.s_prefix+= "_"
    self.save_folder = self.sql_eng.data_folder
    self.out_file = os.path.join(self.save_folder, 
                                 self.s_prefix + __lib__+"_result_data.csv")
    self.log_file = os.path.join(self.save_folder, 
                                 self.s_prefix + __lib__+"_log.txt")
    self._logger("LOGfile: {}".format(self.log_file[:30]))
    return  

  def _load_config(self, str_file = 'gdcb_config.txt'):
      """
      Load JSON configuration file
      """
      
      cfg_file = open(str_file)
      config_data = json.load(cfg_file) 
      return
  

if __name__ =="__main__":
  
  RUN_UPLOAD = True
  
  explorer = GDCBExplorer()
  if RUN_UPLOAD:
    df = pd.read_csv("../data/mode01_codes.csv", encoding = "ISO-8859-1")  
    explorer.sql_eng.OverwriteTable(df,"codes_v1")