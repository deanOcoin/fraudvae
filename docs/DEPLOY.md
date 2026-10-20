# Hosting the demonstration

The app has a *public mode* for running as a website anyone can reach: the same UI and
scoring API as on a laptop, with everything that trains, builds data or talks to a GPU box
removed, no visitor state, and limits on how much scoring one address can ask for.
`deploy/` has everything a small VPS needs; this page says what the mode does and how the
site was put up.

## What public mode is

`FRAUDVAE_PUBLIC=1` (`fraudvae/server/public.py`) changes the server in five ways:

- **No remote-training API and no Compute tab.** `api_remote` is not mounted at all - it is a
  remote shell with the app's own key - and the UI hides the tab and blocks its route.
- **Nothing that changes the server.** `/api/train/local`, `/api/data/{fetch,build,prepare}`,
  loading a different default run and clearing the customer store return 403 with a pointer
  to the repository. A visitor can pick any generation per request; the default stays.
- **No visitor state.** `customer_id` and `remember` are ignored on `/api/score`: histories
  must come with the request, the customer store is never written, no transaction is kept.
- **Rate limits.** A token bucket per client address (behind nginx's own `limit_req`, which
  is coarser and runs first): 60 scores a minute with a burst of 20, 4 batch uploads a minute
  of at most 5,000 rows each with two running at once server-wide, a comparison across at
  most 6 generations per request. 429 with `Retry-After` past that. On top of the buckets a
  global ceiling: four transactions scoring at once across every visitor, then a 503 with
  `Retry-After` after a three-second wait. All of it is in `deploy/env.example`.
- **Fewer details in the metadata.** `/api/meta` drops the host name, file paths and the
  kernel version; a run's metrics drop where it was fetched from and the training process id;
  `remote.json` never leaves the machine that trained a run (`fraudvae models pack` excludes
  it); the API reference lists only the endpoints a visitor can call; a scoring error says
  the request could not be scored and no more.

Everything else - the examples (from `data/examples/`, the redistributable sources only; the
build's own samples file is never read in public mode even if it is on the disk), the
receipts, the model timeline, the comparison strip, the batch upload, the story pages -
works as it does on a laptop.

## The site

[finfraud.dev](https://www.finfraud.dev). Every tab is also a host - `score.`, `data.`,
`models.`, `define.` (how it works), `story.` (the development process) and `api.` (the
API docs) - served by the same app: `app.js` opens the tab that matches the host and
the navigation moves between hosts; nginx holds one certificate for all of them
(`install.sh --domain` requests it for every name that resolves to the server, and
adds the rest on a later run).

## The server

The site runs on a VPS at 1984.hosting (Iceland): 6 vCPU, 16 GB of RAM, 320 GB SSD, no
GPU. Inference is CPU work here; a single transaction scores in about 0.2 s on the v5
generation, a six-way comparison in about a second once every generation is resident
(`FRAUDVAE_PRELOAD=milestones` loads them at start; six generations take ~7 GB). The whole
corpus is not on the server, and neither is the processed data: the only rows it holds are
the example files committed with the repository.

Layout on the box:

```
/srv/fraudvae/app      the checkout (+ .venv)         user fraudvae, read-only to the service
/srv/fraudvae/runs     the pretrained generations     one directory per run, as `models pull` leaves them
/srv/fraudvae/home     FRAUDVAE_HOME: the server log  the one path the service may write
/etc/fraudvae/env      settings (deploy/env.example)
```

## Putting it up

On a fresh Debian 12 / Ubuntu 22.04 server, as root, with the DNS record for the domain
already pointing at it:

```bash
# 1. the code - either a checkout you copied up, or the public repository
scp -r ~/fraudvae root@SERVER:/root/fraudvae         # or let install.sh git clone it

# 2. the pretrained generations - either copied from a machine that has runs/, or pulled
scp -r ~/fraudvae/runs root@SERVER:/srv/fraudvae/runs  # ~4 GB with the paysim graphs; or skip and let install.sh pull the bundles

# 3. everything else
ssh root@SERVER 'bash /root/fraudvae/deploy/install.sh --source /root/fraudvae --domain fraudvae.example.org --email you@example.org'

# 4. once you have logged in with a key: no more passwords over ssh
ssh root@SERVER 'bash /srv/fraudvae/app/deploy/install.sh --source /root/fraudvae --domain fraudvae.example.org --email you@example.org --harden-ssh'
```

`install.sh` is idempotent: it installs the packages, makes the `fraudvae` user and the
directories, copies the code, builds the virtualenv with the CPU build of torch, pulls any
generation that is missing from `/srv/fraudvae/runs`, writes `/etc/fraudvae/env` if there is
none, installs the systemd unit and the nginx site, opens 80/443 and rate-limits 22 in ufw,
gets a Let's Encrypt certificate with a renewal hook that reloads nginx, and sets up
fail2ban, unattended upgrades and log rotation. Afterwards:

```bash
systemctl status fraudvae            # the app (uvicorn on 127.0.0.1:8765)
journalctl -u fraudvae -f            # its log
tail -f /srv/fraudvae/home/server.log
bash /srv/fraudvae/app/deploy/update.sh            # after `git pull` / with --source for a copied checkout
```

The systemd unit runs the service as an unprivileged user with a read-only view of the
system, a 12 GB memory ceiling, no capabilities, a system-call allow-list and no network
beyond localhost; nginx terminates TLS, caps request bodies at 6 MB, limits requests per
address, adds the security headers and answers only for the site's own names. fail2ban
watches sshd and nginx's rate limiter, security updates install themselves, the access log
is anonymised and kept a week. There is no database, no cookie and no session: the site has
nothing to leak that is not already public. `SECURITY.md` at the root of the repository goes
through all of it.

Two things the installer leaves to you. `--harden-ssh` turns password logins off and limits
root to key authentication; it runs only when root already has an authorized key, so make
sure you can log in with yours before you ask for it. And when the kernel has been updated,
`/var/run/reboot-required` exists and the box wants a reboot at a quiet moment (the service
comes back on its own).

### Cloudflare in front, or not

Not needed for a demonstration: the host includes DDoS filtering, nginx and the app each
rate-limit, and the most expensive thing a visitor can ask for is a 5,000-row batch
(seconds of CPU). If the site ever draws real traffic, a Cloudflare proxy in front costs
nothing and adds caching of the static files and a bot filter; the only change is that
`X-Forwarded-For` then carries Cloudflare's chain and `real_ip_header CF-Connecting-IP`
belongs in the nginx config so the per-address limits see the visitor, not Cloudflare.

### Administering it from somewhere that cannot ssh in

The GPU box was driven through a relay during training (`docs/REMOTE_TRAINING.md`): the
machine polls a private git repository for command scripts and pushes the output back,
which works from anywhere that can reach github.com over https. `deploy/relay_bootstrap.sh`
installs the same agent on the server as a systemd service, with a deploy key that can write
to that one repository and nothing else. It was used to run the installation above and is
disabled once the site is up (`systemctl disable --now fraudvae-relay`); the scripts in
`deploy/` are what it ran, so the site needs no such channel afterwards.

## Running the public mode on your own machine

`FRAUDVAE_PUBLIC=1 fraudvae serve` gives the locked-down app locally - useful to see what the
site can and cannot do, or to put the demonstration on an intranet. Without the variable the
same app is the full workbench (training, data, remote GPU).
