# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 12:11:57 2018

@author: Andrei Ionut Damian

@modified:
  2018-04-13 modified load for debug 

"""

class AnomalyPersistor:
  def __init__(self):
    return
  
  def _save_model(self, anomaly_obj):
    """
    saves model status to server
    """
    # save self.features
    s_sql = ''
    for key in anomaly_obj.features.keys():
      for param in anomaly_obj.features[key].keys():
        val = anomaly_obj.features[key][param]
        s_sql+=" begin tran"
        s_sql+="  if exists (select * from dbo.Models with (updlock,serializable) where CarID={} AND Model='{}' AND Categ ='FEATURES' AND Field='{}' AND Param ='{}')".format(
            anomaly_obj.entity_name,
            anomaly_obj.model_name,
            key,
            param
          )
        s_sql+="      begin"
        s_sql+="         update dbo.Models set Value = {}".format(val)
        s_sql+="         where CarID={} AND Model='{}' AND Categ='FEATURES' AND Field='{}' AND Param='{}' ".format(anomaly_obj.entity_name, anomaly_obj.model_name, key, param)
        s_sql+="      end"
        s_sql+="  else"
        s_sql+="     begin"
        s_sql+="         insert into dbo.Models ([CarID], [Model],[Categ],[Field],[Param],[Value])"
        s_sql+="         values ({},'{}','FEATURES','{}','{}',{})".format(anomaly_obj.entity_name, anomaly_obj.model_name, key, param, val)
        s_sql+="     end"
        s_sql+=" commit tran "

    # save self.overall
    for param in anomaly_obj.overall.keys():
        val = anomaly_obj.overall[param]
        s_sql+=" begin tran"
        s_sql+="  if exists (select * from dbo.Models with (updlock,serializable) where CarID={} AND Model='{}' AND Categ ='OVERALL' AND Field='' AND Param ='{}')".format(
            anomaly_obj.entity_name,
            anomaly_obj.model_name,
            param
          )
        s_sql+="      begin"
        s_sql+="         update dbo.Models set Value = {}".format(val)
        s_sql+="         where CarID={} AND Model='{}' AND Categ='OVERALL' AND Field='' AND Param='{}' ".format(anomaly_obj.entity_name, anomaly_obj.model_name, param)
        s_sql+="      end"
        s_sql+="  else"
        s_sql+="     begin"
        s_sql+="         insert into dbo.Models ([CarID], [Model],[Categ],[Field],[Param],[Value])"
        s_sql+="         values ({},'{}','OVERALL','','{}',{})".format(anomaly_obj.entity_name, anomaly_obj.model_name, param, val)
        s_sql+="     end"
        s_sql+=" commit tran "

    anomaly_obj.logger.VerboseLog('Saving features and summary from EntityID={}...'.format(anomaly_obj.entity_name))
    if anomaly_obj.sql_eng.ExecInsert(s_sql):
      anomaly_obj.logger.VerboseLog('Done saving features and summary from EntityID={}.'.format(anomaly_obj.entity_name), 
                                    show_time=True)
    else:
      anomaly_obj.logger.VerboseLog('ERROR saving features and summary from EntityID={}.'.format(anomaly_obj.entity_name), 
                                    show_time=True)      
    return
  
  def _load_model(self, anomaly_obj):
    """
    loads model status from server
    """
    loaded = False
    select_query = "SELECT * FROM Models WHERE CarID={} AND Model='{}'".format(
        anomaly_obj.entity_name, anomaly_obj.model_name) 
    model_df = anomaly_obj.sql_eng.Select(select_query)
    
    if model_df is None:
      anomaly_obj.P("Error finding Model {} for Car with ID {}".format(
          anomaly_obj.model_name, anomaly_obj.entity_name))
      return loaded
    
    _loaded = set()
    _skipped = set()
    
    for _, row in model_df.iterrows():
      category  = row['Categ']
      field     = row['Field']
      param     = row['Param']
      value     = row['Value']
      
      if category.upper() == "FEATURES":
        if field.upper() in anomaly_obj.PRED_NAMES:
          anomaly_obj.features[field][param] = value
          _loaded.add(field)
          loaded = True
        else:
          _skipped.add(field)
      elif category.upper() == "OVERALL":
        anomaly_obj.overall[param] = value

    anomaly_obj.P("Loaded values for {} and skipped {} for EntityID {}".format(
        _loaded, _skipped, anomaly_obj.entity_name))
    
    return loaded
    
    