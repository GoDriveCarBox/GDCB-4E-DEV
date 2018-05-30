from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import CarsHelper
import json
from django.shortcuts import render, redirect
import logging
from .gdcb_explore import GDCBExplorer
from .forms import SearchForm
import pandas as pd
import numpy as np
from math import pi
from . import urls

logger = logging.getLogger(__name__)

df_search = None

crt_user   = None
crt_passwd = None
def map(request):
    return render(request, "api/map.html")

@csrf_exempt
def index(request):
  global crt_user
  global crt_passwd
  global err_msg

  email_field = 'Adresa de email'
  pass_field = 'Parola'
  if request.method == 'POST':
    crt_user = request.POST['email']
    crt_passwd = request.POST['pwd']

    select_id_query = 'SELECT ID FROM Users WHERE Adresa_email =' + "'" + str(crt_user) + "'" + ' AND Parola =' + \
                  "'" + str(crt_passwd) + "'" + ";"
    id_df = urls.gdcb.sql_eng.Select(select_id_query)
    id_user = id_df['ID'][0]
    print("========== ID: ==========")
    print(id_user)
    select_query = 'SELECT * FROM Users WHERE ID =' + "'" + str(id_user) +"'" + ";"
    login_df = urls.gdcb.sql_eng.Select(select_query)
    err_msg = ""
    global nume
    nume_df = pd.DataFrame(login_df['Nume'])
    nume = nume_df['Nume'][0]
    global prenume
    prenume_df = login_df['Prenume']
    prenume = prenume_df[0]
    global tel
    tel_df = login_df['Telefon']
    tel = tel_df[0]
    global descriere
    descriere_df = login_df['Descriere_companie']
    descriere = descriere_df[0]
    global rol
    rol_df = login_df['Rol_user']
    rol = rol_df[0]
    global nume_companie
    nume_comp_df = login_df['Nume_companie']
    nume_companie = nume_comp_df[0]

    if len(login_df):

      print(login_df)
      print("\n")
      print(nume)
      return redirect('/explore')
    else:
      err_msg = "Username sau parola gresita"
      return HttpResponse(err_msg)

  #context = {
#    'error_msg': err_msg,
 # }
  if request.method == 'GET':
    return render(request, 'api/first_page.html')#, context)
    #logger.info("Accessing index view")


def api_view(request):
  if request.method == 'GET':
    df = urls.gdcb.df_predictors.loc[urls.gdcb.df_predictors['Enabled'] == 1]
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
  global df_search
  form = None
  if request.method == 'POST':
    if urls.df_rawdata_toshow is not None:
      logger.info("Searching in RawTable...{}".format(list(urls.df_rawdata_toshow.columns)))
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
        df_search = urls.df_rawdata_toshow.loc[urls.df_rawdata_toshow['CarID']==search_parameters[0]]
        CarID_completed = True
      if search_parameters[1] is not '':
        Code_completed = True
        if CarID_completed is True:
          df_search = df_search.loc[df_search['Code']==search_parameters[1]]
        else:
          df_search = urls.df_rawdata_toshow.loc[urls.df_rawdata_toshow['Code']==search_parameters[1]]

      if (not CarID_completed) and (not Code_completed):
        df_search = pd.DataFrame(columns=list(urls.df_rawdata_toshow.columns))

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

@csrf_exempt
def admin(request):
    global crt_user
    global crt_passwd
    global nume
    global prenume
    global tel
    global descriere
    global df_users
    global rol
    global df_users_insert
    df_users =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["USERS_TABLE"], caching=False)
    #df_users_insert =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["USERS_TABLE"], caching=False)
    df_users_insert =  urls.gdcb.sql_eng.GetEmptyTable(urls.gdcb.config_data["USERS_TABLE"])
    del df_users_insert['ID']




    if request.method == 'GET':
        if crt_user == "" or crt_user == None:
            return redirect("/")
        else:
            print ("DATAFRAME before append:\n")
            print (df_users_insert)
            context = {
                'username': crt_user,
                'passwd': crt_passwd,
                'nume': nume,
                'prenume': prenume,
                'tel': tel,
                'descriere': descriere,
                'rol': rol,
                'users_list': [tuple(x) for x in df_users.to_records(index=False)],
            }
            return render(request, "api/admin.html", context)
    else:
        if crt_user == "" or crt_user == None:
            return redirect("/")
        else:
            context = {
                'username': crt_user,
                'passwd': crt_passwd,
                'nume': nume,
                'prenume': prenume,
                'tel': tel,
                'descriere': descriere,
                'rol': rol,
                'users_list': [tuple(x) for x in df_users.to_records(index=False)],
            }
            row = [request.POST['create_name'], request.POST['create_prenume'], request.POST['create_tel'], request.POST['create_username'], request.POST['create_pass'], request.POST['create_company_type'], request.POST['create_company_name'], request.POST['create_company_desc'], request.POST['create_rol'], 1]
            row_serie = pd.Series(row, index=['Nume', 'Prenume',  'Telefon', 'Adresa_email', 'Parola', 'Tip_cont', 'Nume_companie', 'Descriere_companie', 'Rol_user', 'Flota_detinuta'])
            #df_users_insert.append(row_serie, ignore_index=True)
            print ("DATAFRAME after append:\n")
            df_users_insert = df_users_insert.append(row_serie, ignore_index=True)
            print (df_users_insert.head())
            #sql_insert_query = "INSERT INTO Users  ('Nume', 'Prenume', 'Telefon', 'Adresa_email', 'Parola', 'Tip_cont', 'Nume_companie', 'Descriere_companie', 'Rol_user', 'Flota_detinuta') values ('" +  nume_creat + "','" + prenume_creat + "','" + telefon_creat + "','" + user_creat + "','" + pass_creat + "','" + comp_type_creat + "','" + comp_name_creat + "','" + comp_desc_creat + "','" + rol_user_creat + "'" + ");"
            #print ("SQL:" + sql_insert_query)
            #urls.gdcb.sql_eng.ExecInsert(sql_insert_query)
            print("BEFORE ReadTable\n")
            print()
            urls.gdcb.sql_eng.SaveTable(df_users_insert, urls.gdcb.config_data["USERS_TABLE"])
            return render(request, "api/admin.html", context)
@csrf_exempt
def profile(request):
    global crt_user
    global crt_passwd
    global nume
    global prenume
    global tel
    global descriere
    global nume_companie
    global df_users
    global df_users_update
    df_users_update =  urls.gdcb.sql_eng.GetEmptyTable(urls.gdcb.config_data["USERS_TABLE"])
    id_user = request.GET.get('id')

    del df_users_update['ID']
    df_users =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["USERS_TABLE"], caching=False)
    '''context = {
        'username': crt_user,
        'passwd': crt_passwd,
        'nume': nume,
        'prenume': prenume,
        'tel': tel,
        'rol': rol,
        'descriere': descriere,
        'nume_companie': nume_companie,
        'users_list': [tuple(x) for x in df_users.to_records(index=False)],
    }'''
    if request.method == 'GET':
        if crt_user == "" or crt_user == None:
            return redirect("/")
        elif (request.GET.get('id') is not None):
                    id_user = str(request.GET.get('id'))
                    print("========== ID profil vizitat: ==========")
                    print(id_user)
                    select_query = 'SELECT * FROM Users WHERE ID =' + "'" + id_user +"'" + ";"
                    login_df = urls.gdcb.sql_eng.Select(select_query)
                    err_msg = ""

                    nume_df = pd.DataFrame(login_df['Nume'])
                    nume = nume_df['Nume'][0]

                    prenume_df = login_df['Prenume']
                    prenume = prenume_df[0]

                    tel_df = login_df['Telefon']
                    tel = tel_df[0]

                    descriere_df = login_df['Descriere_companie']
                    descriere = descriere_df[0]

                    rol_df = login_df['Rol_user']
                    rol = rol_df[0]

                    nume_comp_df = login_df['Nume_companie']
                    nume_companie = nume_comp_df[0]

                    context = {
                        'username': crt_user,
                        'passwd': crt_passwd,
                        'nume': nume,
                        'prenume': prenume,
                        'tel': tel,
                        'rol': rol,
                        'descriere': descriere,
                        'nume_companie': nume_companie,
                        'users_list': [tuple(x) for x in df_users.to_records(index=False)],
                    }
                    return render(request, "api/profile.html", context)
        else:
            return render(request, "api/profile.html")

    if request.method == 'POST':
        if request.POST.get("admin_profileform") is not None:
            nume = request.POST['profile_name']
            prenume = request.POST['profile_prenume']
            tel = request.POST['profile_tel']
            company_type = request.POST['profile_company_type']
            company_name = request.POST['profile_company_name']
            descriere = request.POST['profile_company_desc']
            user_rol = request.POST['profile_rol_user']
            id_user = request.POST.get('id')
            row = [nume, prenume, tel, request.POST['create_username'], request.POST['create_pass'], company_type, company_name, company_desc, user_rol, 1]
            row_serie = pd.Series(row, index=['Nume', 'Prenume',  'Telefon', 'Adresa_email', 'Parola', 'Tip_cont', 'Nume_companie', 'Descriere_companie', 'Rol_user', 'Flota_detinuta'])

            update_query = 'UPDATE Users SET Nume =' + nume + 'Prenume = ' + prenume + 'Telefon =' + tel + 'Tip_cont =' + company_type + 'Descriere_companie =' + descriere + ' WHERE ID=' + "'" + str(id_user) + "'" + ";"
            context = {
                'username': crt_user,
                'passwd': crt_passwd,
                'nume': nume,
                'prenume': prenume,
                'tel': tel,
                'rol': rol,
                'descriere': descriere,
                'nume_companie': nume_companie,
                'users_list': [tuple(x) for x in df_users.to_records(index=False)],
            }
            print("Test")
            urls.gdcb.sql_eng.ExecUpdate(update_query)
            if crt_user == "" or crt_user == None:
                return redirect("/")
            else:
                return render(request, "api/profile.html", context)

        if request.POST.get("profileform") is not None:
                nume = request.POST['profile_name']
                prenume = request.POST['profile_prenume']
                tel = request.POST['profile_tel']
                company_type = request.POST['profile_company_type']
                descriere = request.POST['profile_company_desc']
                rol = 'companie'
                nume_companie = 'SRL'
                update_query = 'UPDATE Users SET Nume =' + "'" + nume + "'" + ', Prenume = ' + "'" + prenume + "'" + ', Telefon =' + "'" + tel + "'" + ', Tip_cont =' + "'" + company_type + "'" + ', Descriere_companie =' + "'" + descriere + "'" + ' WHERE Adresa_email =' + "'" + str(crt_user) + "'" + ' AND Parola =' + \
                              "'" + str(crt_passwd) + "'" + ";"
                context = {
                    'username': crt_user,
                    'passwd': crt_passwd,
                    'nume': nume,
                    'prenume': prenume,
                    'tel': tel,
                    'rol': rol,
                    'descriere': descriere,
                    'nume_companie': nume_companie,
                    'users_list': [tuple(x) for x in df_users.to_records(index=False)],
                }
                print(update_query)
                urls.gdcb.sql_eng.ExecInsert(update_query)
                if crt_user == "" or crt_user == None:
                    return redirect("/")
                else:
                    return render(request, "api/profile.html", context)

def test_view(request):
  global crt_user
  global crt_passwd
  from datetime import timedelta


  print(crt_user, crt_passwd)

  #rawdata_view(request)

  df_test = None

  if request.method == 'GET':

    if crt_user == "" or crt_user == None:
      return redirect("/")

    matrix = np.ones((10, 100)) * (-1) # must create dynamic array at least (better update with Ajax)


    for _, row in urls.df_carsxaccounts.iterrows():
      car = row['Masina']
      flota = row['FlotaID']

      matrix[flota, car] = car

    matrix = pd.DataFrame(matrix)
    df_accounts =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["ACCOUNTS_TABLE"], caching=False)
    context = {
      'cars_list': [tuple(x) for x in matrix.to_records(index=False)],
      'account_list': [tuple(x) for x in df_accounts.to_records(index=False)],
      'codes_list': [tuple(x) for x in urls.df_codes.to_records(index=False)],
      'username': crt_user,
      'passwd': crt_passwd,
      'nume': nume,
      'prenume': prenume,
      'tel': tel,
      'rol': rol,
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

    Code = Code[-2:]

    df_test = urls.df_rawdata_toshow.loc[urls.df_rawdata_toshow['CarID']==int(CarID)]
    df_test = df_test.loc[df_test['Code']==Code]
    df_test = df_test[df_test['TimeStamp'] >= start_date]
    df_test = df_test[df_test['TimeStamp'] <= end_date + timedelta(days=1)]
    df_test.index = df_test['TimeStamp']

    if len(df_test) == 0:
      return HttpResponse("Nu s-au gasit date conform selectiilor.")

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
  global df_accounts
  global df_cars
  global df_codes

  logger.info("Accessing rawdata view")
  page = request.GET.get('page', False)
  if page is False:
    logger.info("Updating urls.df_rawdata_toshow")
    df_rawdata_toshow = urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["RAWDATA_TABLE"],\
                                               caching=False)
    df_accounts =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["ACCOUNTS_TABLE"], caching=False)
    df_cars =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["CARS_TABLE"], caching=False)
    df_codes =  urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["PREDICTOR_TABLE"], caching=False)
    df_carsxcodes = urls.gdcb.sql_eng.ReadTable(urls.gdcb.config_data["CARSXCODESV2_TABLE"], caching=False)
    df_carsxaccounts = gdcb.df_carsxaccounts


  if not urls.df_rawdata_toshow is None:
    urls.gdcb.AssociateCodeDescriptionColumns(urls.df_rawdata_toshow)
    urls.gdcb._logger("Rawdata copy: {}".format(list(urls.df_rawdata_toshow.columns)))
    context = {
      'entry_list': [tuple(x) for x in urls.df_rawdata_toshow.to_records(index=False)],
      'page_template': page_template,
    }
    if request.is_ajax():
      template = page_template
    return render(request, template, context)
  return HttpResponse("Internal error")
