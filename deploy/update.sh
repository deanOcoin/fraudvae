#!/usr/bin/env bash
# Update a deployed site to the current code: pull (or rsync from a checkout), reinstall,
# restart. Run as root:  bash /srv/fraudvae/app/deploy/update.sh [--source /path/to/checkout]
set -euo pipefail
APP=/srv/fraudvae/app
SRC=""
[ "${1:-}" = "--source" ] && SRC="$2"
if [ -n "$SRC" ]; then
  rsync -a --delete --exclude .git --exclude .venv --exclude 'data/raw' --exclude 'data/processed' --exclude 'data/prepared' \
        --exclude runs --exclude '*.log' --exclude .env "$SRC"/ "$APP"/
else
  git -C "$APP" pull --ff-only
fi
[ -f /etc/fraudvae/pip-constraints.txt ] && export PIP_CONSTRAINT=/etc/fraudvae/pip-constraints.txt
"$APP/.venv/bin/pip" install -q -e "$APP"
chown -R root:fraudvae "$APP"
chmod -R g-w,o-rwx "$APP"
# nginx config may have changed too. install.sh records the names in /etc/fraudvae/site.conf;
# an older installation is read off the certificate path in the live file
DOMAIN=""; NAMES=""; APPNAMES=""
[ -f /etc/fraudvae/site.conf ] && . /etc/fraudvae/site.conf
if [ -z "$DOMAIN" ] && [ -f /etc/nginx/sites-available/fraudvae ]; then
  DOMAIN=$(grep -m1 -o '/etc/letsencrypt/live/[^/]*/fullchain.pem' /etc/nginx/sites-available/fraudvae | cut -d/ -f5 || true)
fi
if [ -n "$DOMAIN" ] && [ -z "$NAMES" ]; then
  NAMES="$DOMAIN"; APPNAMES=""
  for s in www score data models define story api; do NAMES="$NAMES $s.$DOMAIN"; APPNAMES="$APPNAMES $s.$DOMAIN"; done
  APPNAMES="${APPNAMES# }"
fi
if [ -n "$DOMAIN" ] && [ -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
  sed -e "s/APPNAMES/$APPNAMES/g" -e "s/NAMES/$NAMES/g" -e "s/DOMAIN/$DOMAIN/g" "$APP/deploy/nginx.conf" \
    | sed "s|    # ssl_certificate / ssl_certificate_key: written by certbot|    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;\n    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|" \
    > /etc/nginx/sites-available/fraudvae.new
  if ! cmp -s /etc/nginx/sites-available/fraudvae.new /etc/nginx/sites-available/fraudvae; then
    mv /etc/nginx/sites-available/fraudvae.new /etc/nginx/sites-available/fraudvae
    mkdir -p /var/cache/nginx/fraudvae && chown www-data:www-data /var/cache/nginx/fraudvae
    nginx -t && systemctl reload nginx
  else
    rm -f /etc/nginx/sites-available/fraudvae.new
  fi
fi
install -m 644 "$APP/deploy/fraudvae.service" /etc/systemd/system/fraudvae.service
systemctl daemon-reload
systemctl restart fraudvae
# the answers nginx cached came from the old code
rm -rf /var/cache/nginx/fraudvae/* 2>/dev/null || true
sleep 3
systemctl --no-pager --lines=3 status fraudvae || true
curl -s -m 5 http://127.0.0.1:8765/api/meta | head -c 200; echo
