"""账号服务函数。"""

from __future__ import annotations

import hashlib


def normalize_email(email: str) -> str:
    """邮箱规范化后再哈希，避免大小写和空格导致无法找回。"""

    return email.strip().lower()


def email_sha256(email: str) -> str:
    """计算邮箱 SHA256。"""

    return hashlib.sha256(normalize_email(email).encode("utf-8")).hexdigest()
