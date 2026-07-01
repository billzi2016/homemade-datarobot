"""accounts 应用路由。"""

from __future__ import annotations

from django.urls import path

from accounts import views


urlpatterns = [
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("password-reset/", views.password_reset_view, name="password-reset"),
]
