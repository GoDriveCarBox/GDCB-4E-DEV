from django.http import HttpResponse
from .utils import CarsHelper
import json
from django.shortcuts import render
import logging
from .gdcb_explore import GDCBExplorer
from .forms import SearchForm
import pandas as pd

logger = logging.getLogger(__name__)

def index(request):
    #logger.info("Accessing index view")
    return render(request, 'api/index.html')

df_rawdata_toshow = None
df_search = None
gdcb = GDCBExplorer()

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
  df_test = None
  if request.method == 'GET':
    return render(request, 'api/date_pick.html')

  if request.method == 'POST':
    print(request.POST)
    CarID = request.POST['CarID']
    Code = request.POST['Code']
    start_date = request.POST['start_date']
    end_date = request.POST['end_date']
    chart_type = request.POST['charttype']
    group_by = request.POST['groupdatesby']

    if (CarID is '') or (Code is '') or (start_date is '') or (end_date is ''):
      return HttpResponse("Internal error - Empty parameters")

    from datetime import datetime
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')
    
    df_test = df_rawdata_toshow.loc[df_rawdata_toshow['CarID']==int(CarID)]
    df_test = df_test.loc[df_test['Code']==Code]
    df_test = df_test[df_test['TimeStamp'] >= start_date]
    df_test = df_test[df_test['TimeStamp'] <= end_date]
    df_test.index = df_test['TimeStamp']

    df_grouped = df_test[['TimeStamp', 'ViewVal']].resample(group_by).mean().fillna(0)

    group_by_dict = {
      'D': 'Day',
      'M': 'Month',
      'W': 'Week',
      'A': 'Year'
    }

    if chart_type == 'bar':
      plt_title = 'CarID: {}, Code: {}, Range: {}-{}, GroupBy: {}s'.format(CarID, Code, start_date.strftime("%B %d, %Y"), end_date.strftime("%B %d, %Y"), group_by_dict[group_by])
    else:
      plt_title = 'CarID: {}, Code: {}, Range: {}-{}'.format(CarID, Code, start_date, end_date)


    create_barchart(df_grouped, plt_title, group_by_dict[group_by])
    return render(request, 'api/barchart.html')


def create_barchart(df_grouped, plt_title, xaxis_title):
  import plotly.plotly as py
  import plotly.graph_objs as go
  import plotly
  from plotly.offline import plot
  import os

  labels = list(df_grouped.index.values)
  labels = list(map(lambda x: pd.to_datetime(x).strftime("%b %d, %Y"), labels))
  values = df_grouped['ViewVal'].tolist()

  trace1 = go.Bar(x=labels, y=values, marker=dict(
          color='rgb(34,196,234))',
          line=dict(
              color='rgb(0,188,255)',
              width=1.5,
          )
      ),
      opacity=0.6)
  data = [trace1]
  layout = go.Layout(
      title=plt_title,
      xaxis=dict(
          tickangle=-45,
          title=xaxis_title,
          titlefont=dict(
              size=16,
              color='rgb(107, 107, 107)'
          ),
          tickfont=dict(
              size=14,
              color='rgb(107, 107, 107)'
          )
      ),
      yaxis=dict(
          title='Mean Value',
          titlefont=dict(
              size=16,
              color='rgb(107, 107, 107)'
          ),
          tickfont=dict(
              size=14,
              color='rgb(107, 107, 107)'
          )
      ))
  fig = go.Figure(data=data, layout=layout)

  filename = os.path.dirname(os.path.abspath(__file__))
  templates_dir_name = os.path.join(filename, "templates/api/")
  filename = os.path.join(templates_dir_name, "barchart.html")
  plot(fig, config=dict(displayModeBar=False, showLink=False), filename=filename, auto_open=False)


def rawdata_view(request, template='api/rawdata.html', page_template='api/rawdata_page.html'):
  global df_rawdata_toshow
  global gdcb
  logger.info("Accessing rawdata view")
  page = request.GET.get('page', False)
  if page is False:
    logger.info("Updating df_rawdata_toshow")
    gdcb = GDCBExplorer()
    df_rawdata_toshow = gdcb.sql_eng.ReadTable(gdcb.config_data["RAWDATA_TABLE"],\
                                               caching=False)
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
