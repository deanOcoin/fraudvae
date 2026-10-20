#!/usr/bin/env bash
# Set up the public demonstration on a fresh Debian 12 / Ubuntu 22.04+ server, as root:
#
#   bash deploy/install.sh --domain fraudvae.example.org [--source /path/to/checkout | --git URL] [--email you@example.org | --no-email]
#
# What it does, in order, and all of it again without harm when re-run:
#   1. packages: python3 + venv, nginx, certbot, git, ufw
#   2. a system user `fraudvae` and /srv/fraudvae/{app,runs,home}
#   3. the code into /srv/fraudvae/app (rsync of a checkout you scp'd up, or a git clone)
#   4. a virtualenv with the cpu build of torch (no 2 GB cuda download) and the app
#   5. the pretrained generations: whatever is already in /srv/fraudvae/runs is kept; anything
#      missing is pulled from the github release (`fraudvae models pull --all`)
#   6. /etc/fraudvae/env from deploy/env.example (kept if it exists), the systemd unit, nginx,
#      the firewall (22, 80, 443), and a certificate when --domain and --email are given
#
# The app never listens on a public port itself; nginx does, with rate limits in front of the
# app's own. Re-run after `git pull` is what deploy/update.sh does.
set -euo pipefail

DOMAIN=""; SRC=""; GIT_URL="https://github.com/deanOcoin/fraudvae"; EMAIL=""; NO_CERT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --source) SRC="$2"; shift 2 ;;
    --git) GIT_URL="$2"; shift 2 ;;
    --email) EMAIL="$2"; shift 2 ;;
    --no-email) EMAIL="-"; shift ;;        # a certificate without an account e-mail (no expiry notices)
    --no-cert) NO_CERT=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

APP=/srv/fraudvae/app
RUNS=/srv/fraudvae/runs
HOME_DIR=/srv/fraudvae/home
log() { printf '\n== %s\n' "$*"; }

log "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip nginx certbot python3-certbot-nginx git rsync ufw curl libgomp1 dnsutils >/dev/null

log "user and directories"
id fraudvae >/dev/null 2>&1 || useradd --system --home-dir /srv/fraudvae --shell /usr/sbin/nologin fraudvae
mkdir -p "$APP" "$RUNS" "$HOME_DIR" /etc/fraudvae

log "code -> $APP"
if [ -n "$SRC" ]; then
  rsync -a --delete --exclude .git --exclude .venv --exclude 'data/raw' --exclude 'data/processed' --exclude 'data/prepared' \
        --exclude runs --exclude '*.log' --exclude .env "$SRC"/ "$APP"/
elif [ -d "$APP/.git" ]; then
  git -C "$APP" pull --ff-only
else
  git clone --depth 1 "$GIT_URL" "$APP"
fi

log "virtualenv"
# a vps whose virtual cpu hides the x86-64-v2 instructions (no sse4.2 / popcnt: "QEMU Virtual CPU
# version 2.5+" on some hosts) cannot load the current numpy wheels, which require them; the
# last numpy / scipy wheels built for the plain x86-64 baseline still can, so pin those there.
# torch's cpu wheel and lightgbm have a baseline code path and run either way (more slowly)
if ! grep -q -m1 -w sse4_2 /proc/cpuinfo; then
  echo "cpu without sse4.2 ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)): pinning numpy 2.2 / scipy 1.15 (baseline x86-64 wheels)"
  printf 'numpy==2.2.6\nscipy==1.15.3\n' > /etc/fraudvae/pip-constraints.txt
  export PIP_CONSTRAINT=/etc/fraudvae/pip-constraints.txt
fi
if [ ! -x "$APP/.venv/bin/python" ]; then
  python3 -m venv "$APP/.venv"
fi
"$APP/.venv/bin/pip" install -q --upgrade pip wheel
# torch's cpu wheel: a tenth of the size of the default cuda one, and the only one a vps can use
"$APP/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$APP/.venv/bin/pip" install -q -e "$APP"
"$APP/.venv/bin/python" -c 'import fraudvae, torch, lightgbm; print("fraudvae", fraudvae.__version__, "torch", torch.__version__, "lightgbm", lightgbm.__version__)'

# the app makes these at start-up; under the unit's read-only view of /srv/fraudvae/app they must already exist
mkdir -p "$APP/data/raw" "$APP/data/processed" "$APP/data/prepared"

log "pretrained generations in $RUNS"
present=$(ls -d "$RUNS"/*/model.pt 2>/dev/null | wc -l || true)
echo "$present generation(s) already here"
if [ "$present" -lt 6 ]; then
  echo "pulling the rest from the github release (large: ~1.1 GB)"
  FRAUDVAE_RUNS_DIR="$RUNS" "$APP/.venv/bin/python" -m fraudvae.cli models pull --all || echo "pull failed; scp the runs/ directory into $RUNS and re-run"
fi
# nothing a visitor must not see travels with a run: the box it was fetched from, pid files
find "$RUNS" -maxdepth 2 \( -name remote.json -o -name '*.pid' -o -name launcher.log \) -delete 2>/dev/null || true

log "settings, unit, nginx"
[ -f /etc/fraudvae/env ] || cp "$APP/deploy/env.example" /etc/fraudvae/env
chown -R fraudvae:fraudvae /srv/fraudvae
chmod 750 /srv/fraudvae/home
install -m 644 "$APP/deploy/fraudvae.service" /etc/systemd/system/fraudvae.service
systemctl daemon-reload
systemctl enable fraudvae >/dev/null
systemctl restart fraudvae

# every tab of the site is also a subdomain (app.js routes by host); an ip gets no subdomains
SUBS="www score data models define story api"
if printf '%s' "$DOMAIN" | grep -Eq '^[0-9.]+$'; then
  NAMES="$DOMAIN"; APPNAMES="$DOMAIN"; CERT_ARGS=""
else
  NAMES="$DOMAIN"; APPNAMES=""; CERT_ARGS="-d $DOMAIN"
  for s in $SUBS; do NAMES="$NAMES $s.$DOMAIN"; APPNAMES="$APPNAMES $s.$DOMAIN"; CERT_ARGS="$CERT_ARGS -d $s.$DOMAIN"; done
  APPNAMES="${APPNAMES# }"
fi
render_nginx() {
  sed -e "s/APPNAMES/$APPNAMES/g" -e "s/NAMES/$NAMES/g" -e "s/DOMAIN/$DOMAIN/g" "$APP/deploy/nginx.conf" \
    | sed "s|    # ssl_certificate / ssl_certificate_key: written by certbot|    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;\n    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|"
}
if [ -n "$DOMAIN" ]; then
  sed -e "s/APPNAMES/$APPNAMES/g" -e "s/NAMES/$NAMES/g" -e "s/DOMAIN/$DOMAIN/g" "$APP/deploy/nginx.conf" > /etc/nginx/sites-available/fraudvae
  mkdir -p /var/www/html
  if [ -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    render_nginx 1 > /etc/nginx/sites-available/fraudvae
  else
    # no certificate yet: serve plain http until certbot has run, so its challenge can be answered
    python3 - "$DOMAIN" <<'EOF'
import re, sys
p = "/etc/nginx/sites-available/fraudvae"
s = open(p).read()
# drop the https server blocks and let the http one proxy instead of redirecting
s = re.sub(r"\n# the bare domain is the www host\nserver \{\n    listen 443 ssl.*", "\n", s, flags=re.S)
s = s.replace("    location / { return 301 https://$host$request_uri; }",
              "    location / { proxy_pass http://fraudvae_app; proxy_set_header Host $host; proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; proxy_set_header X-Real-IP $remote_addr; }")
open(p, "w").write(s)
EOF
  fi
  ln -sf /etc/nginx/sites-available/fraudvae /etc/nginx/sites-enabled/fraudvae
  rm -f /etc/nginx/sites-enabled/default
  nginx -t && systemctl reload nginx
  if [ "$NO_CERT" = 0 ] && [ -n "$EMAIL" ] && [ ! -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    log "certificate for $DOMAIN"
    # one certificate for the domain and every subdomain (each validated over http)
    if [ "$EMAIL" = "-" ]; then MAIL_ARGS="--register-unsafely-without-email"; else MAIL_ARGS="-m $EMAIL"; fi
    # only the names that already point here go on the certificate; a missing record would fail
    # the whole request. Re-run once the record exists and --expand adds the name
    MYIP=$(curl -s -m 5 https://api.ipify.org || hostname -I | awk '{print $1}')
    LIVE_ARGS=""
    for n in $NAMES; do
      if dig +short "$n" @1.1.1.1 | grep -qx "$MYIP"; then LIVE_ARGS="$LIVE_ARGS -d $n"; else echo "  $n does not resolve to $MYIP yet: left off the certificate for now"; fi
    done
    CERT_ARGS="$LIVE_ARGS"
    certbot certonly --webroot -w /var/www/html $CERT_ARGS --cert-name "$DOMAIN" --non-interactive --agree-tos $MAIL_ARGS --expand \
      || echo "certbot failed (dns not pointing here yet for one of the names?); re-run install.sh once it does"
    if [ -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
      render_nginx 1 > /etc/nginx/sites-available/fraudvae
      nginx -t && systemctl reload nginx
    fi
  fi
else
  echo "no --domain given: nginx not configured; the app listens on 127.0.0.1:8765 only"
fi

log "firewall"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null
ufw status | head -8

log "done"
sleep 3
systemctl --no-pager --lines=5 status fraudvae || true
curl -s -m 5 http://127.0.0.1:8765/api/meta | head -c 300 || true; echo
