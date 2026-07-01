"""账号页面视图。"""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth import get_user_model
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from accounts.forms import LoginForm, PasswordResetVerifyForm, RegisterForm
from accounts.models import UserProfile
from accounts.services import email_sha256


@require_http_methods(["GET", "POST"])
def register_view(request):
    """注册用户。邮箱只保存 SHA256。"""

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            User = get_user_model()
            user = User.objects.create_user(
                username=form.cleaned_data["username"],
                password=form.cleaned_data["password"],
            )
            UserProfile.objects.create(user=user, email_sha256=email_sha256(form.cleaned_data["email"]))
            login(request, user)
            messages.success(request, "注册完成。")
            return redirect("task-list")
    else:
        form = RegisterForm()
    return render(request, "accounts/register.html", {"form": form})


@require_http_methods(["GET", "POST"])
def login_view(request):
    """登录用户。"""

    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            login(request, form.cleaned_data["user"])
            messages.success(request, "登录完成。")
            return redirect("task-list")
    else:
        form = LoginForm()
    return render(request, "accounts/login.html", {"form": form})


def logout_view(request):
    """退出登录。"""

    logout(request)
    messages.success(request, "已退出。")
    return redirect("login")


@require_http_methods(["GET", "POST"])
def password_reset_view(request):
    """
    本地密码重置。

    当前没有邮件发送系统，因此流程是：
    用户输入用户名 + 邮箱，系统用邮箱 SHA256 匹配后允许重设密码。
    """

    if request.method == "POST":
        form = PasswordResetVerifyForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data["user"]
            user.set_password(form.cleaned_data["new_password"])
            user.save()
            messages.success(request, "密码已重置，请重新登录。")
            return redirect("login")
    else:
        form = PasswordResetVerifyForm()
    return render(request, "accounts/password_reset.html", {"form": form})
