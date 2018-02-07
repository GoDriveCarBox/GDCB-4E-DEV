"""
Definition of views.
"""

from django.shortcuts import render
from django.http import HttpRequest
from django.template import RequestContext
from datetime import datetime
from django.http import HttpResponse
import requests, json
import urllib



def home(request):
    
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )

def contact(request):
    
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )

    
def testpage(request):
    return render(request, 'app/testpage.html',
                  {
                      'title':'testpage',
                      'message':'TEST'}
                  )


def BodyWS(seach_city):
    return """<?xml version="1.0" encoding="utf-8"?>
                        <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
                            <soap:Body>
                                <GetCitiesByCountry xmlns="http://www.webserviceX.NET">
                                    <CountryName>"""+seach_city+"""</CountryName>
                            </GetCitiesByCountry>
                        </soap:Body>
                    </soap:Envelope>"""

def search(request):
    if request.method == 'POST':
        search_city = request.POST.get('textfield', None)
        try:     
            #in partea de jos comentata am incercat sa folosesc city_id si sa incarc json-ul
            #din pacate, nu pot incarca un json mai mare de 5 MB si asa ca am facut unul mic de test pentru a functiona si l-am incarcat in proiect
            '''
            city_list = json.loads(open('test.json').read())
            print(city_list["maps"][0]["country"])

            for i in range(len(city_list['maps'])):
                if search_city== city_list['maps'][i]["name"]:
                    city_id=city_list['maps'][i]["id"]

            #print(city_id)
            ''' 
            #am folosit direct numele orasului si a mers
            url = 'http://api.openweathermap.org/data/2.5/weather?q='+search_city+'&appid=b366e0224583d4806aa6be6735bb1616'
            
            r=requests.get(url)

            data= r.json()

            if len(data)>2:
                stringdata = "<h1> Teperatura in {0} este de {1:.2f} grade C <h1>".format(search_city, float(data['main']['temp'])- 273.15)
            else:
                stringdata = "<h1> Orasul {0} nu a fost gasit".format(search_city)
            
            return HttpResponse(stringdata)
        except Exception: 
            #print(Exception.args)
            raise
    else:
        return render(request, 'testpage.html')

def index(request):
    return render(request, 'testpage.html')
