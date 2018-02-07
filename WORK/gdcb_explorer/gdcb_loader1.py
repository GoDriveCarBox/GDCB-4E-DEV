# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:57:39 2017

@author: Andrei
"""

from gdcb_explore import GDCBExplorer
import pandas as pd


if __name__=="__main__":
  gdcb = GDCBExplorer()

  df_cars = gdcb.df_cars[["ID"]]
  df_codes = gdcb.df_predictors[["Code", "ID"]]
  df_cars.columns = ['CarID']
  df_codes.columns = ['Code', 'CodeID']
  df_codes["key"] = 1
  df_cars["key"] = 1

  df = pd.merge(df_cars,df_codes, on="key")
  df.drop("key", axis = 1,  inplace = True)
  df["ID"] = list(range(len(df)))
  df = df[["ID", "CarID", "Code", "CodeID"]]

  #gdcb.sql_eng.OverwriteTable(df, "CarsXCodes")