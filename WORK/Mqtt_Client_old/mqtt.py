import paho.mqtt.client as mqtt
from mqtt_utils import LoadLogger
import os
import pandas as pd
import datetime
from time import time as tm
from gdcb_explore import GDCBExplorer
from threading import Timer


class GMqttClient():

  def __init__(self, name = "GMqtt1", config_file = "config.txt", debug = False):
    self.init_cache_timestamp = 0
    self.init_gdcb_timestamp = 0
    self.name = name
    self.config_file = config_file
    self.logger = LoadLogger(self.name, self.config_file)
    self.config_data = self.logger.config_data
    self._init_gdcb_explorer()
    self._get_config_data()

    self.df_crt_batch = pd.DataFrame(columns = self.cols_list)
    self.ct = 0
    self.num_received_msg = 0
    self.DEBUG = debug
    self.DEBUG_COUNTER = 1
    self.minutes_alive = 0
    self.flag_connected = False
    self.recv_msg_in_interval = True
    self._init_cache()
    self.print_status_service_run()

  def print_status_service_run(self):
    if self.flag_connected and recv_msg_in_interval:
      self.timer = Timer(self.check_time, self.print_status_service_run)
      self.timer.start()
      self.log("I am alive and listening on topic {}, server {} for {:.2f} hours".
        format(self.topic_tree[:-2], self.server, self.minutes_alive))
      self.minutes_alive += self.check_time
      self.recv_msg_in_interval = False
    else:
      self.timer = Timer(self.check_time, self.print_status_service_run)
      self.timer.start()

  def _get_config_data(self):
    self.base_folder  = self.config_data['BASE_FOLDER']
    self.app_folder   = self.config_data['APP_FOLDER']
    self.server       = self.config_data['SERVER']
    self.port         = int(self.config_data['PORT'])
    self.keep_alive   = int(self.config_data['KEEP_ALIVE'])
    self.topic_token  = self.config_data['TOPIC_PIDS']
    self.topic_tree   = self.topic_token + "/#"
    self.batch_size   = int(self.config_data['BATCH_SIZE'])
    self.path_tokens  = self.config_data['PATH']
    self.dbg_tokens   = self.config_data['DEBUG_CODES']
    self.h_init_cache = self.config_data['HOURS_REINIT_CACHE']
    self.h_init_gdcb  = self.config_data['HOURS_REINIT_GDCB']
    self.check_time   = float(self.config_data['MINUTES_CHECK_INTERVAL'])
    self.cols_list    = self.path_tokens + [self.gdcb.raw_sval_field, self.gdcb.raw_time_field]
    
    self.valid_topics = [self.topic_token]
    return

  def _init_gdcb_explorer(self):
    t_sec = tm()
    if (self.init_gdcb_timestamp == 0) or ((t_sec - self.init_gdcb_timestamp) >= self.h_init_gdcb * 3600):
      self.gdcb = GDCBExplorer(self.logger)
      self.init_gdcb_timestamp = tm()
    return

  def _init_cache(self):
    t_sec = tm()
    if (self.init_cache_timestamp == 0) or ((t_sec - self.init_cache_timestamp) >= self.h_init_cache * 3600):
      str_query = 'SELECT B.CarID, A.VIN, B.Code, B.CodeID, C.Mult, C.[Add], C.Units  FROM ' +\
                  self.gdcb.config_data["CARS_TABLE"] + " A, " +\
                  self.gdcb.config_data["CARSXCODES_TABLE"] + " B, " +\
                  self.gdcb.config_data["PREDICTOR_TABLE"] + " C " +\
                  ' WHERE A.ID = B.CarID AND B.CodeID = C.ID'
  
      self.df_cache = self.gdcb.sql_eng.Select(str_query)
      self.logger.SaveDataframe(self.df_cache, fn = 'DATA_CACHE')
      self.init_cache_timestamp = tm()
    return
  
  def log(self, msg, show_time =  False):
    self.logger.VerboseLog(msg, show_time = show_time)

  def _on_connect(self, client, userdata, flags, rc):

    self.log("Connected to server {} on port {} with keepalive {}".format(
      self.server, self.port, self.keep_alive))
    self.log("\t Client: {}".format(client))
    self.log("\t User Data: {}".format(userdata))
    self.log("\t Flags: {}".format(flags))
    self.log("\t Result code: {}".format(rc))

    client.subscribe(self.topic_tree)
    self.flag_connected = True

    return

  def _on_disconnect(self, client, userdata, rc):
    if rc != 0:
      self.log("Client disconnected unexpectedly. Disconnection code = {}.".format(rc))
    else:
      self.log("Client disconnected.")
    return

  def _on_message(self, client, userdata, msg):

    self.num_received_msg += 1
    self.time_since_last_msg = 0
    if self.DEBUG and (self.num_received_msg >= self.DEBUG_COUNTER):
      self.client.disconnect()

    self.log("Received mesage: Topic=[{}]; Payload=[{}]; Tstmp=[{}]; Index=[{}]".format(
        msg.topic, msg.payload, msg.timestamp, self.num_received_msg))

    if msg.retain == 1:
      self.log("Message skipped due to retain=1")
      return
    self.register_message(msg.topic, msg.payload)
    return

  def _on_subscribe(self, client, userdata, mid, granted_qos):

    self.log("Subscribe on {} accepted:".format(self.topic_tree))
    self.log("\t Client: {}".format(client))
    self.log("\t User Data: {}".format(userdata))
    self.log("\t Mid: {}".format(mid))
    self.log("\t Granted_qos: {}".format(granted_qos))
    
  def DispatchSavedBatch(self, file_path, dispatch = True):
    self.df_loaded_batch = pd.read_csv(file_path)
    self.log("Loaded saved batch [..{}].".format(file_path[-35:]))
    """
    if dispatch:
      self._dispatch(self.df_loaded_batch, save_to_disk=False)
    """
    return

  def register_message(self, topic_data, payload_data):
    topic_values = [data for data in topic_data.split('/') if data != ""]
    if topic_values[0] in self.valid_topics:
      msg_dict = {}
      for i in range(1,len(topic_values)):
        msg_dict[self.path_tokens[i-1]] = topic_values[i]
      msg_dict[self.cols_list[-2]] = payload_data.decode("utf-8")
      recv_tmstp = datetime.datetime.now()
      recv_tmstp = recv_tmstp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
      msg_dict[self.cols_list[-1]] = recv_tmstp
  
      self.df_crt_batch = self.df_crt_batch.append(pd.DataFrame(msg_dict, index=[0]), 
        ignore_index = True)
      
      if self.df_crt_batch.shape[0] >= self.batch_size:
        self._dispatch(self.df_crt_batch)
        self.df_crt_batch = self.df_crt_batch[0:0]
    return

  def _dispatch(self, df, save_to_disk=True):

    df_to_dispatch = df.copy()

    filename  = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    filename += "_" + str(self.batch_size) + "batch.csv"

    if os.path.isfile(filename):
      filename  = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
      filename += "_" + str(self.ct)
      filename += "_" + str(self.batch_size) + "batch.csv"
      self.ct += 1
    else:
      self.ct = 0       
    
    self._init_gdcb_explorer()    
    self._init_cache()
    """
    if save_to_disk:
      df_to_dispatch.to_csv(filename, float_format='{:f}'.format, encoding='utf-8',
                            date_format='%d-%m-%Y %H-%M-%S', index=False)
    """
    self.write_to_database(df_to_dispatch)
    df_to_dispatch = df_to_dispatch[0:0]
    return

  def write_to_database(self, df):
    df_joined = pd.merge(df, self.df_cache, how='left', on=['VIN', 'Code'])
    df_joined.drop('VIN', axis=1, inplace=True)
    self.gdcb.DumpDfToRawData(df_joined)
    return
    

  def setup_connection(self):

    self.client = mqtt.Client()
    self.client.on_connect    = self._on_connect
    self.client.on_message    = self._on_message
    self.client.on_subscribe  = self._on_subscribe
    self.client.on_disconnect = self._on_disconnect

    self.client.connect(self.server, self.port, self.keep_alive)

    self.client.loop_forever()

if __name__ == "__main__":

  gdc = GMqttClient()
  #gdc.DispatchSavedBatch('batches/29-01-2018_21-50-56_100batch.csv')
  #gdc.write_to_database(gdc.df_loaded_batch)
  gdc.setup_connection()
