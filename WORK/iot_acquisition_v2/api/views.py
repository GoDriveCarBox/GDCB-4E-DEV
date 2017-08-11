from django.http import HttpResponse
from .utils import CarsHelper, gdcb
import json
from django.shortcuts import render
import logging

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'api/index.html')

def api_view(request):
    if request.method == 'GET':
        return HttpResponse('Please make a POST request')

    if request.method == 'POST':
        if request.META['CONTENT_TYPE'] == 'application/json':
            helper = CarsHelper()
            response = helper.get_cars(request)
            return HttpResponse(json.dumps(response), content_type="application/json")
        response = {}
        response['status'] = 'BAD_REQUEST'
        response['status_code'] = '400'
        response['description'] = 'Please send a JSON object'
        return HttpResponse(json.dumps(response), content_type="application/json")

def rawdata_view(request):
    print(request.GET.get('page'))
    df_rawdata_toshow = gdcb.sql_eng.ReadTable(gdcb.config_data["RAWDATA_TABLE"], caching=False)
    if not df_rawdata_toshow is None:
      gdcb._logger("Rawdata copy: {}".format(list(df_rawdata_toshow.columns)))
    return render(request, 'api/rawdata.html', {'df': df_rawdata_toshow.tail(1000)})