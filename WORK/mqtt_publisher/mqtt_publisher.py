import paho.mqtt.client as mqtt
from mqtt_utils import LoadLogger
import random


class MQTTPublisher():
  def __init__(self, name = "MQTTPublisher", config_file = "config.txt"):
    self.logger = LoadLogger(lib_name=name,
                             config_file=config_file,
                             log_suffix="",
                             TF_KERAS=False,
                             HTML=True)
    self._init_config_data()
    self._init_connection()


  def _init_config_data(self):
    self.server       = self.logger.config_data['SERVER']
    self.port         = int(self.logger.config_data['PORT'])
    self.keepalive    = int(self.logger.config_data['KEEP_ALIVE'])
    self.topic_tree   = self.logger.config_data['TOPIC']
    self.vin_to_send  = self.logger.config_data['TEST_VIN']
    self.dev_to_send  = self.logger.config_data['TEST_DEV']
    self.code_to_send = self.logger.config_data['TEST_CODE']


  def log(self, str_to_log):
    return self.logger.VerboseLog(str_to_log)


  def _init_connection(self):
    self.client = mqtt.Client()
    self.client.on_connect    = self._on_connect
    self.client.on_subscribe  = self._on_publish
    self.client.on_disconnect = self._on_disconnect
    self.client.connect(self.server, self.port, self.keepalive)


  def _on_connect(self, client, userdata, flags, rc):
    self.log("Connected to server {} on port {} with keepalive {}".format(
      self.server, self.port, self.keep_alive))
    self.log("\t Client: {}".format(client))
    self.log("\t User Data: {}".format(userdata))
    self.log("\t Flags: {}".format(flags))
    self.log("\t Result code: {}".format(rc))

    client.subscribe(self.topic_tree)
    return


  def _on_disconnect(self, client, userdata, rc):
    if rc != 0:
      self.log("Client disconnected unexpectedly. Disconnection code = {}.".format(rc))
    else:
      self.log("Client disconnected.")
    return


  def _on_publish(client, userdata, result):
    self.log("Data published.")
    self.log("\t Client: {}".format(client))
    self.log("\t Userdata: {}".format(userdata))
    self.log("\t Result: {}".format(result))


  def send_messages(self, num_messages):
    topic_list = [self.topic_tree, self.vin_to_send, 
                  self.dev_to_send, self.code_to_send]
    topic_data = "/".join(topic_list) + "/"
    
    for i in range(num_messages):
      payload_data = random.randint(0, 300) / random.randint(1, 6)
      self.client.publish(topic_data, payload_data)

if __name__ == "__main__":
  publisher = MQTTPublisher()
  publisher.send_messages(30000)