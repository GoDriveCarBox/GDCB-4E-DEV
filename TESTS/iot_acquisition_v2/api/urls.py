from django.conf.urls import url
from . import views
from .gdcb_explore import GDCBExplorer
import os

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
    logger_lib = None
    print("Logger library not found in shared repo.")
    #raise Exception("Couldn't find google drive folder!")
  else:
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)), flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)), flush = True)

  return logger_lib

df_rawdata_toshow = None
#df_search = None
gdcb = GDCBExplorer()
df_rawdata_toshow = gdcb.sql_eng.ReadTable(gdcb.config_data["RAWDATA_TABLE"],\
                                               caching=False)

df_accounts =  gdcb.sql_eng.ReadTable(gdcb.config_data["ACCOUNTS_TABLE"], caching=False)
df_cars =  gdcb.sql_eng.ReadTable(gdcb.config_data["CARS_TABLE"], caching=False)
df_codes =  gdcb.sql_eng.ReadTable(gdcb.config_data["PREDICTOR_TABLE"], caching=False)
df_carsxcodes = gdcb.sql_eng.ReadTable(gdcb.config_data["CARSXCODESV2_TABLE"], caching=False)
df_carsxaccounts = gdcb.df_carsxaccounts
if df_rawdata_toshow is not None:
    gdcb.AssociateCodeDescriptionColumns(df_rawdata_toshow)

urlpatterns = [
    url(r'^show/$', views.rawdata_view),
    url(r'^show$', views.rawdata_view),
    url(r'^upload/$', views.api_view),
    url(r'^upload$', views.api_view),
    url(r'^search/$', views.search_view),
    url(r'^search$', views.search_view),
    url(r'^explore$', views.test_view),
    url(r'^explore/$', views.test_view),
    url(r'^map$', views.map),
    url(r'^profile$', views.profile),
    url(r'^profile/$', views.profile),
    url(r'^admin$', views.admin),
    url(r'^admin/$', views.admin, name='admin'),
    url(r'', views.index),
]
