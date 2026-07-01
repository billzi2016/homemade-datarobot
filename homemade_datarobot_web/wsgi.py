"""WSGI 入口。"""

from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homemade_datarobot_web.settings")
application = get_wsgi_application()
