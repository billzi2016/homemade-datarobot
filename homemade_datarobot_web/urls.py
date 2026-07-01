"""Django 顶层路由。"""

from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

from tasks.api import api


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api.urls),
    path("", include("tasks.urls")),
]
