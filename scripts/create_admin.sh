#!/usr/bin/env bash
set -euo pipefail

# 本地开发 admin 初始化脚本。
#
# 账号信息必须来自 .env 或调用时传入的环境变量：
#   DJANGO_ADMIN_USERNAME
#   DJANGO_ADMIN_EMAIL
#   DJANGO_ADMIN_PASSWORD
#
# 示例：
#   cp .env.example .env
#   ./scripts/create_admin.sh

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

: "${DJANGO_ADMIN_USERNAME:?missing DJANGO_ADMIN_USERNAME; set it in .env}"
: "${DJANGO_ADMIN_EMAIL:?missing DJANGO_ADMIN_EMAIL; set it in .env}"
: "${DJANGO_ADMIN_PASSWORD:?missing DJANGO_ADMIN_PASSWORD; set it in .env}"

python3 manage.py migrate

DJANGO_ADMIN_USERNAME="$DJANGO_ADMIN_USERNAME" \
DJANGO_ADMIN_EMAIL="$DJANGO_ADMIN_EMAIL" \
DJANGO_ADMIN_PASSWORD="$DJANGO_ADMIN_PASSWORD" \
python3 manage.py shell -c "
import os
import hashlib
from django.contrib.auth import get_user_model
from accounts.models import UserProfile

User = get_user_model()
username = os.environ['DJANGO_ADMIN_USERNAME']
email = os.environ['DJANGO_ADMIN_EMAIL']
password = os.environ['DJANGO_ADMIN_PASSWORD']

user, _ = User.objects.get_or_create(username=username, defaults={'email': email})
user.email = email
user.is_staff = True
user.is_superuser = True
user.set_password(password)
user.save()
email_digest = hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()
UserProfile.objects.update_or_create(user=user, defaults={'email_sha256': email_digest})

print(f'admin user ready: {username}')
"

echo "Django admin: http://127.0.0.1:18743/admin/"
