"""tasks 应用路由。"""

from __future__ import annotations

from django.urls import path

from tasks import views


urlpatterns = [
    path("", views.task_list_view, name="task-list"),
    path("tasks/create/", views.task_create_view, name="task-create"),
    path("tasks/<str:task_id>/", views.task_detail_view, name="task-detail"),
    path("tasks/<str:task_id>/run/", views.task_run_view, name="task-run"),
    path("tasks/<str:task_id>/events/", views.task_events_view, name="task-events"),
    path("tasks/<str:task_id>/download/", views.task_download_list_view, name="task-download-list"),
    path("tasks/<str:task_id>/download/file/", views.task_file_download_view, name="task-file-download"),
]
