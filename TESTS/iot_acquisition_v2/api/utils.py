
from django.utils.six import BytesIO
from rest_framework.parsers import JSONParser
from .gdcb_explore import GDCBExplorer
import logging
import pandas as pd
from time import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CarsHelper:
    def __init__(self):
      self.gdcb = GDCBExplorer()
      self.searches_process_entry=0
      self.create_raw_entry = 0

    def create_response(self, status, status_code, description):
      self.response['status'].append(status)
      self.response['status_code'].append(status_code)
      self.response['description'].append(description)

    def check_inconsistence(self, data):
       if (len(data['CarID']) != len(data['Code'])) or\
          (len(data['CarID']) != len(data['Value'])) or\
          (len(data['Code']) != len(data['Value'])):

         self.response['status'] = 'BAD_REQUEST'
         self.response['status_code'] = '400'
         self.response['description'] = 'Inconsistent data - request dropped'
         logger.info("Different array lenghts; sending response")
         return True
       return False

    def process_entry(self, carsxcodes_entry, CarID, Code, Value, idx):
      try:
        temptime = time()
        df_predictors_rows = self.gdcb.df_predictors[self.gdcb.code_field]==carsxcodes_entry['CodeID']
        n = self.gdcb.df_predictors.loc[df_predictors_rows].index.tolist()[0]
        self.searches_process_entry += (time()-temptime)
        logger.info("\tFound CodeID {} at position {} in df_pred"
                    .format(carsxcodes_entry['CodeID'], n))

        if not (self.gdcb.df_predictors.loc[n,self.gdcb.active_field]):
          logger.info("\tCodeID {} not enabled".format(carsxcodes_entry['CodeID']))
          self.create_response('BAD_REQUEST', '400', 'Code not enabled')
        else:
          self.create_response('OK', '200', 'Entry registered')
          temptime = time()
          self.gdcb.CreateRawEntry(CarID, carsxcodes_entry['CodeID'], Value, n, idx)
          self.create_raw_entry += (time()-temptime)
          logger.info("\tEntry registered; df_rawdata.size={}".format(len(self.gdcb.raw_codes)))
      except Exception as err:
        logger.error(err)

    def get_cars(self, request):
      total_search_df = 0
      total_process_entry = 0
      start_time = time()
      stream = BytesIO(request.body)
      data = JSONParser().parse(stream)
      logger.info("POST Request at /api/, data={}".format(data))
      self.response = {}
      if self.check_inconsistence(data) is True:
         return self.response

      self.response['status'] = list()
      self.response['status_code'] = list()
      self.response['description'] = list()
      nr_samples = len(data['CarID'])
      self.df = pd.DataFrame(columns = ['CarID', 'CodeID', 'IntValue'])

      self.gdcb.EmptyRawData()
      tuples = list(zip(data['CarID'], data['Code']))


      str_query = 'SELECT * FROM ' + self.gdcb.config_data["CARSXCODESV2_TABLE"] +\
                  ' as T WHERE EXISTS (SELECT * FROM (VALUES ' + ','.join(map(str, tuples)) +\
                  ') AS V(CarID, Code) WHERE T.CarID = V.CarID and T.Code = V.Code)'

      df_carsxcodes_entries = self.gdcb.sql_eng.Select(str_query)
      logger.info("Selected entries from carsxcodes ... size={}; columns={}".format(df_carsxcodes_entries.shape[0], list(df_carsxcodes_entries.columns)))
      

      for i in range(nr_samples):
        logger.info("Checking sample#{}/{}".format(i+1, nr_samples))
        CarID = data['CarID'][i]
        Code = data['Code'][i]
        Value = data['Value'][i]
        car_found_db = True
        code_found_db = True
        try:
          self.gdcb.df_cars.loc[CarID-1]
          logger.info("\tCar %d found in df_cars" % (CarID))
        except Exception as err:
          logger.error(err)
          logger.info("\tCar %d not found in df_cars" % (CarID))
          car_found_db = False

        if self.gdcb.df_predictors[:]['Code'].str.contains(Code).sum() == 0:
          logger.info("\tCode %s not found in df_predictors" % (Code))
          code_found_db = False
        else:
          logger.info("\tCode %s found in df_predictors" % (Code))

        if car_found_db and code_found_db:
          temptime = time()
          aux_df = df_carsxcodes_entries[df_carsxcodes_entries['CarID'] == CarID]
          aux_df = aux_df[aux_df['Code'] == Code]
          total_search_df += (time() - temptime)

          temptime = time()
          self.process_entry(aux_df.iloc[0], CarID, Code, Value, i)
          total_process_entry += (time() - temptime)
        else:
          self.create_response('BAD_REQUEST', '400', 'Car or Code not found in db')

      end_time = time()
      logger.info("Checked {} samples in {}s .. auxdf_search_time={}s .. process_all_entries_time={}s .. searches_all_entries_time={}s .. create_raw_entries_time={}s"
                  .format(nr_samples, end_time-start_time, total_search_df, total_process_entry, self.searches_process_entry, self.create_raw_entry))

      start_time = time()
      self.save_cars_to_db()
      end_time = time()
      logger.info("Saved {} samples to db in {}s".format(nr_samples, end_time-start_time))

      return self.response

    def save_cars_to_db(self):
      logger.info("Trying to save entries to db...")
      self.gdcb.DumpRawData(from_lists=True)
      logger.info("Finished dumping rawdata...")
