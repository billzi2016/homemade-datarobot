"""账号相关表单。"""

from __future__ import annotations

from django import forms
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.password_validation import validate_password

from accounts.models import UserProfile
from accounts.services import email_sha256


class RegisterForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=150)
    email = forms.EmailField(label="邮箱")
    password = forms.CharField(label="密码", widget=forms.PasswordInput)
    password_confirm = forms.CharField(label="确认密码", widget=forms.PasswordInput)

    def clean_username(self) -> str:
        username = self.cleaned_data["username"].strip()
        User = get_user_model()
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("用户名已存在。")
        return username

    def clean_email(self) -> str:
        email = self.cleaned_data["email"]
        digest = email_sha256(email)
        if UserProfile.objects.filter(email_sha256=digest).exists():
            raise forms.ValidationError("该邮箱已被注册。")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")
        if password and password_confirm and password != password_confirm:
            raise forms.ValidationError("两次输入的密码不一致。")
        if password:
            validate_password(password)
        return cleaned_data


class LoginForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=150)
    password = forms.CharField(label="密码", widget=forms.PasswordInput)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password = cleaned_data.get("password")
        if username and password:
            user = authenticate(username=username, password=password)
            if user is None:
                raise forms.ValidationError("用户名或密码错误。")
            cleaned_data["user"] = user
        return cleaned_data


class PasswordResetVerifyForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=150)
    email = forms.EmailField(label="注册邮箱")
    new_password = forms.CharField(label="新密码", widget=forms.PasswordInput)
    new_password_confirm = forms.CharField(label="确认新密码", widget=forms.PasswordInput)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        email = cleaned_data.get("email")
        password = cleaned_data.get("new_password")
        password_confirm = cleaned_data.get("new_password_confirm")

        if password and password_confirm and password != password_confirm:
            raise forms.ValidationError("两次输入的新密码不一致。")
        if password:
            validate_password(password)

        if username and email:
            User = get_user_model()
            try:
                user = User.objects.select_related("profile").get(username=username)
            except User.DoesNotExist as exc:
                raise forms.ValidationError("用户名或邮箱不匹配。") from exc
            if user.profile.email_sha256 != email_sha256(email):
                raise forms.ValidationError("用户名或邮箱不匹配。")
            cleaned_data["user"] = user
        return cleaned_data
