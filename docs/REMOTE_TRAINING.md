# Training on a rented GPU

The full corpus (~42M rows, ~29M of them legitimate training rows) is a few
hours per epoch on a laptop and a couple of minutes per epoch on a modern GPU.
`fraudvae advise` (or the Compute tab) benchmarks the real model on your machine
and extrapolates:

```
$ fraudvae advise --config gpu_full
train rows: 29,183,537  epochs: 30
local (cpu, arm): 9,800 rows/s -> ~49.6 h
recommendation: RENT A GPU
  RTX 3090 / 4070 Ti (24/12 GB)   ~0.87 h  ~$0.17
  RTX 4090 (24 GB)                ~0.57 h  ~$0.20
  A100 40/80 GB                   ~0.37 h  ~$0.34
  ...
```

The GPU rows/s are constants in `fraudvae/remote/advisor.py`. The 4090 row is
measured (434k rows/s on a vast.ai box with the v4 schema, 1.1 min/epoch over
29M rows; 516k on the smaller v3 schema); the others are scaled guesses.
Update them from the `rows_per_sec` field in `metrics.jsonl` after a real
run. The estimate covers the VAE epochs only: the booster adds 10-25 min on
the box's CPUs and calibration + evaluation another ~8 min, and early
stopping usually ends the VAE well before the configured epochs (16 of 30 on
the v4 run), so it roughly evens out.

## What to rent (vast.ai)

- v4 (the common schema only): any CUDA GPU with ≥ 16 GB VRAM, RAM ≥ 32 GB,
  disk ≥ 60 GB. A single RTX 4090 did the shipped v4 run.
- v5 (history encoder, wide blocks, the graph / text / behaviour modules and
  their datasets): **one H100 / H200 class GPU, ≥ 64 GB RAM (128 better), ≥ 16
  cores, ≥ 500 GB disk**. The raw downloads alone are ~60 GB unpacked, the
  prepared arrays plus the sequence index roughly double that, and the
  history transformer over 29M rows is a billion token-steps per epoch. More
  than one GPU does nothing: one model trains at a time.
- 16 GB disk is *not* enough even for v4: the image, the parquet upload and
  the prepared arrays together overflow it and `prepare` dies half way with a
  confusing error.
- image: the "PyTorch" template (`pytorch/pytorch:*-cuda12.*-runtime` or
  vast's `vastai/pytorch`). Anything with a CUDA torch in `python3` works; if
  there is none the provisioner installs one, which is a 3 GB download.
- launch mode: ssh, with **Direct SSH** enabled so the box has a public ip:port
  rather than going through the proxy. User is `root`.
- add the app's public key to your vast.ai account *before* renting; keys are
  injected at instance start.

## Steps in the UI

1. Compute tab → copy the public key. `fraudvae key` prints the same.
2. Rent the instance, copy its `ssh -p PORT root@IP` line, paste it (the
   *Paste ssh line* button reads the clipboard, or just type host/port/user).
3. *Connect & inspect*: pins the host key, shows GPU / RAM / disk / python /
   torch. First connection shows the fingerprint; a changed fingerprint later
   is an error.
4. *Provision server*: creates `~/fraudvae` on the box, installs torch if
   needed and the other requirements, uploads the code as a tarball, uploads
   `data/processed/*.parquet` (skipping files already there at the same size,
   so a dropped connection just resumes), runs `prepare --name full` there.
5. *Launch*: `nohup python -m fraudvae.train --config gpu_full --prepared full
   --name <run>` on the box; the pid is remembered in
   `~/.fraudvae/remote_jobs.json`. One launch does the whole chain: VAE
   epochs on the GPU, then the LightGBM booster on the box's CPUs (a few
   minutes), then calibration (blender + thresholds) and the test evaluation.
6. *Refresh status* (or auto refresh): reads `status.json`, `metrics.jsonl` and
   the log tail over sftp. *Stop* kills the pid.
7. *Fetch results & load model*: downloads `model.pt`, `booster.txt` +
   `booster.json`, calibration, eval, metrics, latent map and log into
   `runs/<run>/`, marks it as fetched from the remote and loads it into the
   scorer.

Uploading the 1 GB of parquet is the slow part (3-4 MB/s from a home
connection, about 5 minutes); `prepare` on the box takes about a minute.

v5 adds a second way to get data onto the box, for the 60 GB of raw downloads
behind the new modules that no laptop should process: tick *push raw
downloads from <folder> and build on the box* (CLI: `--raw
~/Downloads/fraud-data-raw`). The folder is scanned and every file is
identified by its name or by the names inside the zip
(`fraudvae/data/aux_registry.py`), so kaggle's "archive (6).zip" lands as
`data/raw/baf/`. Uploads resume byte-exactly after a dropped connection.
Datasets with a public url (BrainRun on zenodo) and the github sources are
fetched by the box itself, then `build` runs there and `prepare` after it.
Expect the push to take hours on a home uplink; the task log shows per-file
progress and the Compute tab keeps polling.

`fraudvae remote run --host IP --port PORT` does steps 3-7 unattended from a
terminal and polls until the run finishes. `fraudvae remote launch ... --name
RUN --stage post` (or "stage: post only" in the UI) keeps an existing remote
run's VAE and redoes only the booster, calibration and evaluation, e.g. after
changing the booster settings in the config. Provision with "upload data" off
first if the code changed; the box keeps its prepared set. The post process
writes its own pid into `status.json`, which is how the status check knows a
re-launched run is alive (the launch record still has the trainer's pid).

Interrupted instances: the checkpoint is saved after every improving epoch;
relaunch with the same run name and *resume* ticked to continue from it.

## Doing it by hand

```bash
scp -P PORT -r fraudvae configs pyproject.toml requirements.txt scripts root@IP:~/fraudvae/
scp -P PORT data/processed/*.parquet data/processed/stats.json root@IP:~/fraudvae/data/processed/
ssh -p PORT root@IP 'bash ~/fraudvae/scripts/bootstrap_remote.sh'
ssh -p PORT root@IP 'cd ~/fraudvae && python3 -m fraudvae.data.prepare --name full && nohup python3 -m fraudvae.train --config gpu_full --prepared full --name full-gpu > train.out 2>&1 &'
# later
scp -P PORT -r root@IP:~/fraudvae/runs/full-gpu runs/
```

## When the laptop cannot stay awake: the relay

Everything above drives the box over ssh from the laptop, which has to be awake for
every command. For a build that runs for days there is a second channel that needs no
laptop at all: a **private git repository as a mailbox**. `scripts/relay_agent.sh`
runs on the box under `nohup`, pulls the repository every 15 seconds, runs any new
`cmd/<id>.sh` it finds, and pushes the output back as `out/<id>.log` / `.rc` /
`.done`; files dropped under `files/` (code tarballs) reach the box the same way. The
other side is `fraudvae remote relay --repo DIR run "command"` (or the `Relay` class),
which pushes a command and waits for its output. Set-up: create an empty private
repository, generate a key on the box (`ssh-keygen -t ed25519`), add its public half as
a *deploy key with write access* on that repository only, clone it on the box with
that key (`git config core.sshCommand 'ssh -i ~/.ssh/<key>'`), start the agent. The
deploy key is the box's only credential and opens nothing but the relay; delete the
repository when the box is gone. Long jobs are launched detached inside a command
(`nohup ... &`) and watched with later ones; the round trip is a poll interval or two.

## Security notes

- The private key lives in `~/.fraudvae/ssh/id_ed25519` with mode 600 and is
  not in the repo (`.gitignore`). `GET /api/remote/key?rotate=1` makes a new
  one.
- The web server binds to 127.0.0.1 by default. If you expose it, anyone who
  can reach it can use your remote boxes.
- Host keys are trust-on-first-use. If you re-rent and get the same IP with a
  different key, delete the line from `~/.fraudvae/ssh/known_hosts`.
