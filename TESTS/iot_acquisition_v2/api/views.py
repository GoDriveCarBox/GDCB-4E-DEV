from django.http import HttpResponse
from .utils import CarsHelper
import json
from django.shortcuts import render, redirect
import logging
from .gdcb_explore import GDCBExplorer
from .forms import SearchForm
import pandas as pd
import numpy as np
from math import pi

logger = logging.getLogger(__name__)

df_rawdata_toshow = None
df_search = None
gdcb = GDCBExplorer()

crt_user   = None
crt_passwd = None

def index(request):

  global crt_user
  global crt_passwd

  if request.method == 'POST':
    crt_user = request.POST['email']
    crt_passwd = request.POST['pwd']

    select_query = 'SELECT * FROM Users WHERE Email =' + "'" + str(crt_user) + "'" + ' AND Password =' + \
                  "'" + str(crt_passwd) + "'" + ";"

    login_df = gdcb.sql_eng.Select(select_query)

    if len(login_df) != 0:
      print(login_df)
      return redirect('/explore')
    else:
      return HttpResponse("Error")

  if request.method == 'GET':
    return render(request, 'api/first_page.html')
    #logger.info("Accessing index view")

def api_view(request):
  global gdcb
  if request.method == 'GET':
    df = gdcb.df_predictors.loc[gdcb.df_predictors['Enabled'] == 1]
    return render(request, 'api/doc.html', {'df': df})

  if request.method == 'POST':
    if request.META['CONTENT_TYPE'] == 'application/json':
      helper = CarsHelper()
      response = helper.get_cars(request)
      #logger.info("Response: {}".format(response))
      return HttpResponse(json.dumps(response), content_type="application/json")
    response = {}
    response['status'] = 'BAD_REQUEST'
    response['status_code'] = '400'
    response['description'] = 'Please send a JSON object'
    return HttpResponse(json.dumps(response), content_type="application/json")

def search_view(request, template='api/rawdata.html', page_template='api/search_page.html'):
  global df_rawdata_toshow
  global gdcb
  global df_search
  form = None
  if request.method == 'POST':
    if df_rawdata_toshow is not None:
      logger.info("Searching in RawTable...{}".format(list(df_rawdata_toshow.columns)))
    else:
      return HttpResponse("Internal error")
    form = SearchForm(request.POST)

  if form is not None:
    logger.info("Received form parameters in search_view")
    search_parameters = None
    if form.is_valid():
      search_parameters = (form.cleaned_data['CarID'], form.cleaned_data['Code'])

    if search_parameters is None:
      logger.info("search_parameters are empty")
    else:
      logger.info("search_parameters are not empty")
      CarID_completed = False
      Code_completed = False
      if search_parameters[0] is not None:
        df_search = df_rawdata_toshow.loc[df_rawdata_toshow['CarID']==search_parameters[0]]
        CarID_completed = True
      if search_parameters[1] is not '':
        Code_completed = True
        if CarID_completed is True:
          df_search = df_search.loc[df_search['Code']==search_parameters[1]]
        else:
          df_search = df_rawdata_toshow.loc[df_rawdata_toshow['Code']==search_parameters[1]]

      if (not CarID_completed) and (not Code_completed):
        df_search = pd.DataFrame(columns=list(df_rawdata_toshow.columns))

  if not df_search is None:
    logger.info("Dataframe containing search results is not none")
    context = {
      'entry_list': [tuple(x) for x in df_search.to_records(index=False)],
      'page_template': page_template,
    }
    if request.is_ajax():
      template = page_template
    return render(request, template, context)
  return HttpResponse("Internal error")

def test_view(request):
  global df_rawdata_toshow
  global gdcb
  global crt_user
  global crt_passwd
  from datetime import timedelta


  print(crt_user, crt_passwd)

  rawdata_view(request)

  df_test = None

  if request.method == 'GET':

    if crt_user == None:
      return redirect("/")

    matrix = np.ones((10, 100)) * (-1) # must create dynamic array at least (better update with Ajax)


    for _, row in df_carsxaccounts.iterrows():
      car = row['Masina']
      flota = row['FlotaID']

      matrix[flota, car] = car

    matrix = pd.DataFrame(matrix)

    context = {
      'cars_list': [tuple(x) for x in matrix.to_records(index=False)],
      'account_list': [tuple(x) for x in df_accounts.to_records(index=False)],
      'codes_list': [tuple(x) for x in df_codes.to_records(index=False)],
      'username': crt_user,
      'passwd': crt_passwd,
      'page_template': 'api/new_test.html',
    }
    return render(request, 'api/erik_index.html', context)

  if request.method == 'POST':
  
    if request.POST.get("bar_plot") is not None:
      chart_type = "bar"
    elif request.POST.get("line_plot") is not None:
      chart_type = "linie"
    elif request.POST.get("log_out") is not None:
      crt_user, crt_passwd = None, None
      return redirect("/")

    AccountID = request.POST['flota']
    CarID = request.POST['car']
    Code = request.POST['code']

    start_date = request.POST['d_inceput']
    end_date = request.POST['d_sfarsit']

    group_by = request.POST['tip_agregare']

    if (CarID is '') or (Code is '') or (start_date is '') or (end_date is ''):
      return HttpResponse("Internal error - Empty parameters")

    from datetime import datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df_test = df_rawdata_toshow.loc[df_rawdata_toshow['CarID']==int(CarID)]
    df_test = df_test.loc[df_test['Code']==Code]
    df_test = df_test[df_test['TimeStamp'] >= start_date]
    df_test = df_test[df_test['TimeStamp'] <= end_date + timedelta(days=1)]
    df_test.index = df_test['TimeStamp']

    if len(df_test) == 0:
      return HttpResponse("Wrong query")

    if str.lower(group_by) != "fara":
      df_grouped = df_test[['TimeStamp', 'ViewVal']].resample(group_by).mean().fillna(0)
    else:
      df_grouped = df_test

    group_by_dict = {
      'D': 'Zile',
      'W': 'Saptamani',
      'M': 'Luni',
      'A': 'Ani'
    }

    indicator = df_test.iloc[0]['Description']
    if str.lower(group_by) != "fara":
      plt_title = "Analiza de tip {} pentru indicatorul '{}' al masinii {} din flota {} in perioada {}-{} agregat pe {}".format(chart_type, indicator, CarID, AccountID, start_date.strftime("%d.%m.%Y"), end_date.strftime("%d.%m.%Y"), group_by_dict[group_by])
    else:
      plt_title = "Analiza de tip {} pentru indicatorul '{}' al masinii {} din flota {} in perioada {}-{}".format(chart_type, indicator, CarID, AccountID, start_date.strftime("%d.%m.%Y"), end_date.strftime("%d.%m.%Y"))


    if str.lower(chart_type) == "bar":
      create_barchart(df_grouped, plt_title, y_title = indicator, group_by = group_by)
      return render(request, 'api/barchart.html')
    elif str.lower(chart_type) == "linie":
      create_lineplot(df_grouped, plt_title, y_title = indicator, group_by = group_by)
      return render(request, 'api/lineplot.html')
    elif str.lower(chart_type) == "histograma":
      create_histogram(df_grouped, plt_title, indicator)
      return render(request, 'api/histogram.html')
    else:
      return HttpResponse("Not implemented")


def is_date(string):
  from dateutil.parser import parse
  try: 
    parse(string)
    return True
  except ValueError:
    return False

def process_plot_title(plt_title):

  words = plt_title.split()
 
  first_line, second_line = None, None
  split_idx = None
  for i in range(len(words)):
    aux = words[i].split('-')

    if len(aux) != 2:
      continue

    if is_date(aux[0]):
      split_idx = i
      break

  if split_idx is not None:
    first_line = " ".join(words[i] for i in range(split_idx))
    first_line = first_line.replace("in perioada", '')

    second_line = " ".join(words[i] for i in range(split_idx, len(words)))
    second_line = "perioada " + str.lower(second_line)


  return first_line, second_line


def process_plot_data(source_df, group_by=None):

  from  more_itertools import unique_everseen
  import calendar

  min_y = min(source_df['ViewVal'].tolist())
  max_y = max(source_df['ViewVal'].tolist())

  if min_y > 0:
    min_y = 0

  source_df = source_df[source_df['ViewVal'] != 0]
  x = [pd.to_datetime(value) for value in list(source_df.index.values)]
  y = source_df['ViewVal'].tolist()
  y = [float("{:.2f}".format(value)) for value in y ]

  #x.append(x[-1] + timedelta(days=1))


  pairs = zip(x, y)
  sorted_pairs = sorted(pairs, key=lambda x: x[0])
  x, y = zip(*sorted_pairs)
  x = [value.strftime('%d/%m/%Y, %H:%M:%S') for value in x]

  del_idx = []
  n = len(x)
  for i in range(0, n - 1):
    for j in range(i + 1, n):

      if x[i] != x[j]:
        break
      else:
        del_idx.append(j)


  if str.lower(group_by) != "N/A":
  
    zipped_l = [(x[i], y[i]) for i in range(len(x)) if i not in del_idx]
    x, y = zip(*zipped_l)

  if group_by == "D":
    x = [value.split(",")[0] for value in x]
  elif group_by == "W":
    x = [value.split(",")[0] for value in x]
  elif group_by == "M":
    x = [calendar.month_name[int(value.split(",")[0].split("/")[1])] for value in x]
  elif group_by == "A":
    x = [value.split(",")[0].split("/")[2] for value in x]

  #[ (l[i], ll[i]) for i in range(len(ll)) if i not in del_idx ] 

  #x = list(unique_everseen(x))

  return x, y, min_y, max_y

def genereta_plot(source_df,  plt_title, x_title = "Data", y_title = None, type= "bar", group_by= None):

  from bokeh.plotting import figure, output_file, show, save
  from bokeh.models import ColumnDataSource, ranges, LabelSet, Title, HoverTool, BoxZoomTool, ResetTool, WheelZoomTool, PanTool
  from bokeh.palettes import PuBu
  from datetime import timedelta
  import os

  x, y, min_y, max_y = process_plot_data(source_df, group_by)

  first_line, second_line = process_plot_title(plt_title)

  source = ColumnDataSource(dict(x= x, y= y))
  hover = HoverTool(tooltips=[
    ("(Data - Valoare)", "(@x - @y{10.2f})"),
  ])
  plot = figure(x_axis_label = x_title, y_axis_label = y_title, x_range = source.data["x"], 
        y_range= ranges.Range1d(start = min_y , end= max_y + 0.1 * max_y), background_fill_color="#d3d3d3", 
        tools = [hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])
        #, output_backend="webgl")

  if first_line is None:
    plot.add_layout(Title(text= plt_title, text_font_size="10pt"), 'above')
  else:
    plot.add_layout(Title(text= second_line, text_font_size="9pt"), 'above')
    plot.add_layout(Title(text= first_line, text_font_size="10pt"), 'above')
       
  plot.sizing_mode = 'scale_height'

  if len(x) > 10:
    plot.xaxis.major_label_orientation = pi/4 + pi/6
  else:
    plot.xaxis.major_label_orientation = pi/4

  plot.xaxis.axis_label_text_font_size = "12pt"
  plot.yaxis.axis_label_text_font_size = "12pt"
  plot.xaxis.major_label_text_font_size = "10pt"
  plot.yaxis.major_label_text_font_size = "10pt"

  if len(x) > 50:
    plot.xaxis.visible = False

  if len(x) < 10:
    labels = LabelSet(x='x', y='y', text='y', level='glyph',
        x_offset=-12.5, y_offset=5, source=source, render_mode='canvas')

  filename = os.path.dirname(os.path.abspath(__file__))
  templates_dir_name = os.path.join(filename, "templates/api/")

  if type == "bar":
    plot.vbar(source=source, x='x', top='y', bottom=0, width=0.5, color=PuBu[7][2])
    filename = os.path.join(templates_dir_name, "barchart.html")
  elif type == "line":
    plot.line(x=x, y=y, line_width=2, color=PuBu[7][2])
    plot.circle(x=x, y=y, size=5, color="navy", alpha=0.5)
    filename = os.path.join(templates_dir_name, "lineplot.html")
  elif type == "hist":
    hist, edges = np.histogram(y, density=True, bins=50)
    plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
    filename = os.path.join(templates_dir_name, "histogram.html")

  if len(x) < 10:
    plot.add_layout(labels)
  output_file(filename, mode="inline")
  save(plot)


def create_barchart(source_df,  plt_title, x_title = "Data", y_title= None, group_by= None):
  genereta_plot(source_df, plt_title, x_title, y_title, "bar", group_by)

def create_lineplot(source_df,  plt_title, x_title = "Data", y_title= None, group_by= None):
  genereta_plot(source_df, plt_title, x_title, y_title, "line", group_by)

def create_histogram(source_df,  plt_title, y_title = "Valoare medie"):
  from bokeh.plotting import figure, output_file, show, save
  from bokeh.models import ColumnDataSource, ranges, LabelSet, Title, HoverTool, BoxZoomTool, ResetTool, WheelZoomTool, PanTool
  from bokeh.palettes import PuBu
  from datetime import timedelta
  import os

  x, y, min_y, max_y = process_plot_data(source_df, "N/A")
  
  first_line, second_line = process_plot_title(plt_title)

  hover = HoverTool(tooltips=[
    ("(x,y)", "($x, $y)"),
  ])
  plot = figure(x_axis_label = "Valoare", y_axis_label = "Numar", 
          background_fill_color="#d3d3d3", tools = [hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])

  source = ColumnDataSource(dict(x= x, y= y))
  labels = LabelSet(x='x', y='y', text='x', level='glyph',
        x_offset=-12.5, y_offset=5, source=source, render_mode='canvas')

  hist, edges = np.histogram(y, density=False, bins='auto')
  plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
      fill_color=PuBu[7][2], line_color="#036564")

  if first_line is None:
    plot.add_layout(Title(text= plt_title, text_font_size="10pt"), 'above')
  else:
    plot.add_layout(Title(text= second_line, text_font_size="9pt"), 'above')
    plot.add_layout(Title(text= first_line, text_font_size="10pt"), 'above')
       
  plot.sizing_mode = 'scale_height'
  plot.xaxis.major_label_orientation = pi/4
  plot.xaxis.axis_label_text_font_size = "12pt"
  plot.yaxis.axis_label_text_font_size = "12pt"
  plot.xaxis.major_label_text_font_size = "10pt"
  plot.yaxis.major_label_text_font_size = "10pt"

  filename = os.path.dirname(os.path.abspath(__file__))
  templates_dir_name = os.path.join(filename, "templates/api/")
  filename = os.path.join(templates_dir_name, "histogram.html")
  plot.add_layout(labels)
  output_file(filename, mode="inline")
  save(plot)

def rawdata_view(request, template='api/rawdata.html', page_template='api/rawdata_page.html'):
  global df_rawdata_toshow
  global df_accounts
  global df_cars
  global df_codes
  global df_carsxcodes
  global df_carsxaccounts
  global gdcb

  logger.info("Accessing rawdata view")
  page = request.GET.get('page', False)
  if page is False:
    logger.info("Updating df_rawdata_toshow")
    gdcb = GDCBExplorer()
    df_rawdata_toshow = gdcb.sql_eng.ReadTable(gdcb.config_data["RAWDATA_TABLE"],\
                                               caching=False)
    df_accounts =  gdcb.sql_eng.ReadTable(gdcb.config_data["ACCOUNTS_TABLE"], caching=False)
    df_cars =  gdcb.sql_eng.ReadTable(gdcb.config_data["CARS_TABLE"], caching=False)
    df_codes =  gdcb.sql_eng.ReadTable(gdcb.config_data["PREDICTOR_TABLE"], caching=False)
    df_carsxcodes = gdcb.sql_eng.ReadTable(gdcb.config_data["CARSXCODESV2_TABLE"], caching=False)
    df_carsxaccounts = gdcb.df_carsxaccounts


  if not df_rawdata_toshow is None:
    gdcb.AssociateCodeDescriptionColumns(df_rawdata_toshow)
    gdcb._logger("Rawdata copy: {}".format(list(df_rawdata_toshow.columns)))
    context = {
      'entry_list': [tuple(x) for x in df_rawdata_toshow.to_records(index=False)],
      'page_template': page_template,
    }
    if request.is_ajax():
      template = page_template
    return render(request, template, context)
  return HttpResponse("Internal error")
