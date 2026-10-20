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
"$APP/.venv/bin/pip" install -q -e "$APP"
chown -R fraudvae:fraudvae /srv/fraudvae/app
# nginx config may have changed too (DOMAIN is whatever the live file says)
if [ -f /etc/nginx/sites-available/fraudvae ]; then
  dom=$(grep -m1 'server_name' /etc/nginx/sites-available/fraudvae | awk '{print $2}' | tr -d ';')
  if [ -n "$dom" ] && [ -e "/etc/letsencrypt/live/$dom/fullchain.pem" ]; then
    sed "s/DOMAIN/$dom/g" "$APP/deploy/nginx.conf" \
      | sed "s|    # ssl_certificate / ssl_certificate_key: written by certbot|    ssl_certificate /etc/letsencrypt/live/$dom/fullchain.pem;\n    ssl_certificate_key /etc/letsencrypt/live/$dom/privkey.pem;\n    include /etc/letsencrypt/options-ssl-nginx.conf;\n    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;|" \
      > /etc/nginx/sites-available/fraudvae.new
    if ! cmp -s /etc/nginx/sites-available/fraudvae.new /etc/nginx/sites-available/fraudvae; then
      mv /etc/nginx/sites-available/fraudvae.new /etc/nginx/sites-available/fraudvae
      nginx -t && systemctl reload nginx
    else
      rm -f /etc/nginx/sites-available/fraudvae.new
    fi
  fi
fi
install -m 644 "$APP/deploy/fraudvae.service" /etc/systemd/system/fraudvae.service
systemctl daemon-reload
systemctl restart fraudvae
sleep 3
systemctl --no-pager --lines=3 status fraudvae || true
curl -s -m 5 http://127.0.0.1:8765/api/meta | head -c 200; echo
