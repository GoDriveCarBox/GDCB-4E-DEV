from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^show/$', views.rawdata_view),
    url(r'^show$', views.rawdata_view),
    url(r'^upload/$', views.api_view),
    url(r'^upload$', views.api_view),
    url(r'^search/$', views.search_view),
    url(r'^search$', views.search_view),
    url(r'^test$', views.test_view),
    url(r'^test/$', views.test_view),
    url(r'', views.index),
]
