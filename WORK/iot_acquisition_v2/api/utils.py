from django.utils.six import BytesIO
from rest_framework.parsers import JSONParser
from .gdcb_explore import GDCBExplorer
import logging
import pandas as pd

logger = logging.getLogger(__name__)

gdcb = GDCBExplorer()

class CarsHelper:
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

    def process_entry(self, CarID, Code, Value):
      car_found_db = True
      code_found_db = True
      try:
        gdcb.df_cars.loc[CarID-1]
        logger.info("\tCar %d found in db" % (CarID))
      except Exception as err:
        logger.error(err)
        car_found_db = False

      if gdcb.df_predictors[:]['Code'].str.contains(Code).sum() == 0:
        logger.info("\tCode %s not found in df_predictors" % (Code))
        code_found_db = False
      else:
        logger.info("\tCode %s found in db" % (Code))

      if car_found_db and code_found_db:
        self.create_response('OK', '200', 'Entry registered')
        try:
          carsxcodes_entry = gdcb.sql_eng.CustomSelect(gdcb.config_data["CARSXCODESV2_TABLE"], CarID, Code)
          logger.info("\tEntry from CarsXCodes: CarID={}...CodeID={}".format(carsxcodes_entry['CarID'], carsxcodes_entry['CodeID']))
          self.df = self.df.append(pd.Series([CarID, carsxcodes_entry['CodeID'], Value], index=['CarID', 'CodeID', 'IntValue']), ignore_index=True)
          logger.info("\tEntry registered; df.size={}".format(len(self.df)))
        except Exception as err:
          logger.error(err)
      else:
        self.create_response('BAD_REQUEST', '400', 'Car or Code not found in db')

    def get_cars(self, request):
      global gdcb
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
      for i in range(nr_samples):
        logger.info("Checking sample#{}/{}".format(i+1, nr_samples))
        CarID = data['CarID'][i]
        Code = data['Code'][i]
        Value = data['Value'][i]
        self.process_entry(CarID, Code, Value)

      self.save_cars_to_db()

      return self.response

    def save_cars_to_db(self):
      global gdcb
      logger.info("Trying to save entries to db...")
      gdcb.DumpRawDataCustomizedDf(self.df)
      logger.info("Finished dumping rawdata...")