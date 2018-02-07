# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:40:20 2018

@author: LaurentiuP
"""

import pandas as pd
from sqlalchemy import create_engine
import urllib
import time as tm

connstr = 'DRIVER=' + "{ODBC Driver 13 for SQL Server}"
connstr+= ';SERVER=' + "carbox.database.windows.net"
connstr+= ';DATABASE=' + "Carbox"
connstr+= ';UID=' + "carbox@carbox"
connstr+= ';PWD=' + "GDCBnpsf0517"

sql_params = urllib.parse.quote_plus(connstr)
engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_params,
                       connect_args={'connect_timeout': 2})


df = pd.read_csv('upload_data.csv')
dfsize_mb = (df.values.nbytes + df.index.nbytes + df.columns.nbytes) / (1024 * 1024)
print("We try to write in DB a table with sz = [{:.2f} MB] ...".format(dfsize_mb))

t0 = tm.time()
df.to_sql("CarsXCodes",
          engine,
          index = False,
          if_exists = 'replace')
t1 = tm.time()
tsec = t1-t0
print("Wrote table in DB in [{:.2f} s]".format(tsec))