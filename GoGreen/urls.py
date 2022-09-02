from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from GoGreen.dash_apps.finished_apps import temperature
from GoGreen.dash_apps.finished_apps import carbondioxide
from GoGreen.dash_apps.finished_apps import arcticseaice
from GoGreen.dash_apps.finished_apps import iceSheets
from GoGreen.dash_apps.finished_apps import worldmaps
from GoGreen.dash_apps.finished_apps import sealevelrise
from GoGreen.dash_apps.finished_apps import carbonfootprint
from GoGreen.dash_apps.finished_apps import temperatureanalyze
from GoGreen.dash_apps.finished_apps import sealevelanalyze
from GoGreen.dash_apps.finished_apps import greenhousegas
from GoGreen.dash_apps.finished_apps import food
from GoGreen.dash_apps.finished_apps import forestfire


# URLConfiguration
urlpatterns = [
    path('', views.home, name="home"),
    path('co2map/', views.worldmaps, name="co2Map"),
    path('footprintcalculator/', views.footprintcalculator, name="footprintcalculator"),
    path('causes/', views.causes, name="causes"),
    path('causes/greenhousegasses', views.greenhousegas, name="greenhousegases"),
    path('causes/deforestation', views.deforestation, name="deforestation"),
    path('causes/deforestation/<str:pk>/', views.deforestationanalyze, name="deforestationanalyze"),
    path('causes/deforesttation/<str:pk>/', views.deforestationanalyze2, name="deforesttationanalyze"),
    path('causes/food', views.food, name="food"),
    path('effects/', views.effects, name="effects"),
    path('effects/temperature', views.temperature, name="temperature"),
    path('effects/sealevel', views.sealevel, name="sealevel"),
    path('effects/forestfire', views.forestfire, name="forestfire"),
    path('effects/arcticsea', views.arcticsea, name="arcticsea"),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)