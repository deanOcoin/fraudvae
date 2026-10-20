# Security

This page says what the public site exposes, what it does not, and how the pieces are held
in place. It is written for someone deciding whether to trust the site or to run the code
themselves; the exact settings live in `deploy/` and `fraudvae/server/public.py`.

## What the site is

[finfraud.dev](https://www.finfraud.dev) is a read-only demonstration. A visitor can score
transactions against pretrained models, read the statistics of the training corpus, browse
the model generations and the project's history, and call the same scoring API from a
program. That is the whole surface. There is no account, no login, no cookie, no session,
no database and no visitor state: nothing a visitor sends is kept, and nothing on the server
belongs to a visitor.

## What a visitor cannot do

The app has two modes. On a laptop it is a workbench that trains models, builds data, and
drives GPU machines over ssh. The site runs it with `FRAUDVAE_PUBLIC=1`, which removes all
of that before the server starts:

- the remote-training API is not mounted; the Compute tab does not exist
- training, data fetching, building and preparing, changing the default model and the
  customer store answer 403 with a pointer to the repository, and are left out of the API
  reference the site publishes
- `customer_id` and `remember` are ignored on `/api/score`: a history must travel with the
  request, and no transaction is stored
- the metadata drops the host name, file paths, process ids and the kernel version

## How much a visitor can ask for

Three layers, from the outside in:

1. **nginx** rate-limits per address (90 scores a minute, 6 batch uploads, 30 of the large
   answers such as a run's latent map, 300 other API calls, 20 concurrent connections), caps
   a request body at 6 MB, times out slow clients, and keeps the read-only answers in a cache
   for ten minutes so repeated requests for them never reach python.
2. **The app's own buckets** are finer: 60 scores a minute with a burst of 20, 4 batch
   uploads of at most 5,000 rows, a comparison across at most 6 generations. A forged
   `X-Forwarded-For` does not buy a fresh bucket; the address nginx appended is the one that
   counts.
3. **A global ceiling** on scoring: at most four transactions score at once across every
   visitor, and two batches. Past that a request waits up to three seconds and then gets a
   503 with `Retry-After`, so the box answers slowly under load rather than swapping.

Inside one request, every field has a size: at most 600 features, 3,000 wide columns, 256
history rows, 20,000 events per behaviour modality, 20,000 characters of text. A csv is read
for at most one row more than the cap, and refused by its header if it is wider than 3,000
columns. Anything over a limit is a 422 before any model runs.

A run name in a URL is a directory name and nothing else (`[A-Za-z0-9][A-Za-z0-9._-]*`,
at most 100 characters): there is no way to build a path from it. In public mode an error
from the scorer says that the request could not be scored, not which exception where; the
detail goes to the server log.

## What the browser is told

Every script, style, font and library the pages use is served from the site itself
(`fraudvae/web/vendor` holds Chart.js, KaTeX, Swagger UI and the fonts), so the content
security policy is `script-src 'self'`: a `<script>` that somehow reached the page would not
run. The other headers are the usual set - `X-Content-Type-Options`, `X-Frame-Options`,
`Referrer-Policy`, `Permissions-Policy`, HSTS for a year including subdomains, cross-origin
opener and resource policies - and the server answers only for its own host names: a request
for any other name, or for the bare IP address, gets no response (nginx closes the connection
on port 80 and refuses the TLS handshake on 443).

## The server

Ubuntu 24.04 on a small VPS. The app runs under systemd as a system user with no shell, with
a read-only view of the file system except its own log directory, no capabilities, a
system-call allow-list, no access to other processes, devices or kernel interfaces, a 12 GB
memory ceiling, and no network except localhost - it talks to nginx and to nothing else. It
listens on 127.0.0.1 only; nginx is the only process on 80 and 443, and ssh is the only
other open port.

ssh is rate-limited by the firewall and watched by fail2ban (five failures in ten minutes
bans an address for an hour), as is nginx's rate limiter. `deploy/install.sh --harden-ssh`
turns password logins off and limits root to key authentication, and does so only when root
already has an authorized key, so the step cannot lock anyone out. Security updates install
themselves; the certificate renews itself and nginx is reloaded when it does.

The access log keeps the network a request came from, not the address: the last octet of an
IPv4 address and the second half of an IPv6 one are dropped before the line is written, and
the logs are rotated after a week. The app's own log holds no addresses.

## Trusting a model file

A checkpoint is a pickle, and a pickle can carry code. Every checkpoint this project writes
holds tensors, plain containers and numpy arrays, so they are read with torch's
`weights_only` loader, which allows exactly those types and refuses anything else
(`fraudvae/model/loading.py`). The pretrained bundles are verified against the sha256 values
in `fraudvae/pretrained.json` before they are unpacked, and unpacked with the `data` filter
that refuses paths outside the target directory. A run you did not train yourself is still a
run you are choosing to execute code around; pull bundles with `fraudvae models pull`, which
checks them, rather than copying a directory from somewhere.

## Reporting something

If you find a way around any of the above, open an issue on the repository, or if it should
not be public yet, use GitHub's private vulnerability reporting on the repository's Security
tab. The site holds nothing of a visitor's, so there is little to protect but the box itself
and the people using the site; both matter.
