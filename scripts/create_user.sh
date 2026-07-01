#!/usr/bin/env bash
set -euo pipefail

# 本地普通用户初始化脚本。
# 脚本可以进入 git；真实账号、邮箱和密码只放在 .env，.env 不进入 git。
#
# 需要 .env 中存在：
#   DJANGO_APP_USERNAME
#   DJANGO_APP_EMAIL
#   DJANGO_APP_PASSWORD

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

: "${DJANGO_APP_USERNAME:?missing DJANGO_APP_USERNAME; set it in .env}"
: "${DJANGO_APP_EMAIL:?missing DJANGO_APP_EMAIL; set it in .env}"
: "${DJANGO_APP_PASSWORD:?missing DJANGO_APP_PASSWORD; set it in .env}"

python3 manage.py migrate

DJANGO_APP_USERNAME="$DJANGO_APP_USERNAME" \
DJANGO_APP_EMAIL="$DJANGO_APP_EMAIL" \
DJANGO_APP_PASSWORD="$DJANGO_APP_PASSWORD" \
python3 manage.py shell -c "
import hashlib
import os
from django.contrib.auth import get_user_model
from accounts.models import UserProfile

User = get_user_model()
username = os.environ['DJANGO_APP_USERNAME']
email = os.environ['DJANGO_APP_EMAIL']
password = os.environ['DJANGO_APP_PASSWORD']

user, _ = User.objects.get_or_create(username=username)
user.is_staff = False
user.is_superuser = False
user.set_password(password)
user.save()

email_digest = hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()
UserProfile.objects.update_or_create(user=user, defaults={'email_sha256': email_digest})

print(f'app user ready: {username}')
"
