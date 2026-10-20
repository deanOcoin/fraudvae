#!/usr/bin/env bash
# Start the relay agent on a server so it can be administered from a machine that cannot
# open an ssh connection to it (the way the GPU box was driven during training, see
# docs/REMOTE_TRAINING.md): the server polls a private git repository for command scripts
# and pushes their output back. Run as root, once:
#
#   bash relay_bootstrap.sh <owner/relay-repo> <path to the deploy key>
#
# The deploy key is a key with write access to that one repository and nothing else; make it
# with `ssh-keygen -t ed25519 -f relay_deploy_key -N ""` and add the .pub half under the
# relay repository's Settings -> Deploy keys with "allow write". The agent runs as root (it
# installs packages and writes nginx's config); disable it when the site is set up:
#
#   systemctl disable --now fraudvae-relay
set -euo pipefail
REPO="${1:?owner/repo}"
KEY="${2:?path to the deploy key}"
DIR=/root/fraudvae-relay
apt-get install -y -qq git >/dev/null 2>&1 || true
install -d -m 700 /root/.ssh
install -m 600 "$KEY" /root/.ssh/fraudvae_relay
ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null
cat > /root/.ssh/config <<EOF
Host github-fraudvae-relay
  HostName github.com
  User git
  IdentityFile /root/.ssh/fraudvae_relay
  IdentitiesOnly yes
EOF
chmod 600 /root/.ssh/config
if [ ! -d "$DIR/.git" ]; then
  git clone -q "git@github-fraudvae-relay:$REPO.git" "$DIR"
fi
AGENT="$(dirname "$(readlink -f "$0")")/../scripts/relay_agent.sh"
[ -f "$AGENT" ] || AGENT="$DIR/relay_agent.sh"
install -m 755 "$AGENT" /usr/local/bin/fraudvae-relay-agent
cat > /etc/systemd/system/fraudvae-relay.service <<EOF
[Unit]
Description=fraudvae relay agent (administration over a git repository)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/fraudvae-relay-agent $DIR
Restart=always
RestartSec=10
Environment=POLL=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now fraudvae-relay
sleep 2
systemctl --no-pager --lines=3 status fraudvae-relay || true
echo "relay agent up; it polls $REPO every 10 s. Disable with: systemctl disable --now fraudvae-relay"
