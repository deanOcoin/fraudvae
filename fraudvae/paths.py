import os
from pathlib import Path

# the package directory, and the repo root two levels up from this file. In a checkout (an
# editable install, or running straight out of the tree) data/ and runs/ default to the root;
# installed as a wheel the root is site-packages, which is nobody's data directory, so the
# defaults move under the app home instead. Everything that ships with the package (configs,
# the web ui, the manifests) is found relative to the package, which works either way.
PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
IN_CHECKOUT = (ROOT / "pyproject.toml").exists()


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


APP_HOME = Path(os.environ.get("FRAUDVAE_HOME") or (Path.home() / ".fraudvae")).expanduser()
_BASE = ROOT if IN_CHECKOUT else APP_HOME


def _p(env, default):
    raw = os.environ.get(env)
    p = Path(raw).expanduser() if raw else Path(default)
    if not p.is_absolute():
        p = _BASE / p
    return p


DATA_DIR = _p("FRAUDVAE_DATA_DIR", "data")
RAW_DIR = _p("FRAUDVAE_RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = DATA_DIR / "processed"
PREPARED_DIR = DATA_DIR / "prepared"
RUNS_DIR = _p("FRAUDVAE_RUNS_DIR", "runs")
# the server appends its startup lines (and, through faulthandler, the python stack of
# any native crash) here. uvicorn's own output only goes to the terminal, and when a
# crash happens inside torch or openmp that terminal is all you'd have
SERVER_LOG = _p("FRAUDVAE_SERVER_LOG", APP_HOME / "server.log")
WEB_DIR = PKG / "web"
CONFIG_DIR = PKG / "configs"


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, PREPARED_DIR, RUNS_DIR, APP_HOME):
        d.mkdir(parents=True, exist_ok=True)
