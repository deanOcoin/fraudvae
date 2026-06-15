import os
from pathlib import Path

# repo root = two levels up from this file. works for an editable install and for
# running straight out of the checkout, which is what i do 99% of the time
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _p(env, default):
    raw = os.environ.get(env)
    p = Path(raw).expanduser() if raw else Path(default)
    if not p.is_absolute():
        p = ROOT / p
    return p


DATA_DIR = _p("FRAUDVAE_DATA_DIR", "data")
RAW_DIR = _p("FRAUDVAE_RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = DATA_DIR / "processed"
PREPARED_DIR = DATA_DIR / "prepared"
RUNS_DIR = _p("FRAUDVAE_RUNS_DIR", "runs")
APP_HOME = _p("FRAUDVAE_HOME", Path.home() / ".fraudvae")
# the server appends its startup lines (and, through faulthandler, the python stack of
# any native crash) here. uvicorn's own output only goes to the terminal, and when a
# crash happens inside torch or openmp that terminal is all you'd have
SERVER_LOG = _p("FRAUDVAE_SERVER_LOG", APP_HOME / "server.log")
WEB_DIR = ROOT / "fraudvae" / "web"
CONFIG_DIR = ROOT / "configs"


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, PREPARED_DIR, RUNS_DIR, APP_HOME):
        d.mkdir(parents=True, exist_ok=True)
