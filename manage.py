#!/usr/bin/env python3
"""
Django 管理入口。

当前仓库此前只有 MLflow 训练子系统，没有 Web 外壳。
这里补上最小可用的 Django 入口，后续所有 `runserver`、`migrate`
等命令都从这里进入。
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    """标准 Django 入口。"""

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homemade_datarobot_web.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
