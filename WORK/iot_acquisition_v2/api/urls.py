from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^rawdata/$', views.rawdata_view),
    url(r'^rawdata$', views.rawdata_view),
    url(r'^api/$', views.api_view),
    url(r'^api$', views.api_view),
    url(r'', views.index),
]
