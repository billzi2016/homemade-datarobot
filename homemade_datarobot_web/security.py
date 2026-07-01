"""
轻量安全响应头中间件。

当前项目是本地原型，但 Web 壳层已经开始提供上传、启动任务和下载能力，
所以需要先把基础浏览器安全边界加上：
- CSP 限制资源来源，降低 XSS 扩散面
- nosniff 避免浏览器错误猜 MIME
- referrer policy 减少路径泄露
- permissions policy 关闭不需要的浏览器能力
"""

from __future__ import annotations

from django.contrib.auth.views import redirect_to_login
from django.urls import reverse


class SecurityHeadersMiddleware:
    """为所有 Django 响应补基础安全头。"""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response.setdefault("X-Content-Type-Options", "nosniff")
        response.setdefault("X-Frame-Options", "DENY")
        response.setdefault("Referrer-Policy", "same-origin")
        response.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: blob: https://django-ninja.dev; "
                "font-src 'self' data:; "
                "connect-src 'self' http://127.0.0.1:* http://localhost:*; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
        )
        return response


class AuthenticatedApiDocsMiddleware:
    """
    限制 API 文档入口必须登录后访问。

    django-ninja 会自动暴露 Swagger UI 和 OpenAPI JSON；这些页面虽然不是业务写接口，
    但会完整暴露当前 API 的路径、参数和响应结构，所以应该和业务页面使用同一套登录边界。
    """

    PROTECTED_PATHS = {"/api/docs", "/api/openapi.json"}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.rstrip("/") in self.PROTECTED_PATHS and not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path(), login_url=reverse("login"))
        return self.get_response(request)
