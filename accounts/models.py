"""
账号扩展模型。

登录、密码哈希和 session 交给 Django 自带 auth.User。
这里额外保存 email_sha256，满足“邮箱不要明文入库”的要求。
"""

from __future__ import annotations

from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    """用户扩展资料。"""

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    email_sha256 = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"profile:{self.user_id}"
