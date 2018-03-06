from django.conf.urls import url
from . import views

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
    url(r'^admin/$', views.admin),    
    url(r'', views.index),
]
