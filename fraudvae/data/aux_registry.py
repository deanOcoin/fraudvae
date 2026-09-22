"""The v5 module datasets: not transaction sources (those are in registry.py and go through the
common schema), but the data the graph, text and behaviour modules train and validate on.
Each one is identified by the files inside it, so a kaggle zip called "archive (8).zip" dropped
in the raw folder still lands under data/raw/<key>/.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from .registry import SOURCES


@dataclass(frozen=True)
class AuxDataset:
    key: str
    name: str
    module: str                       # graph | text | behaviour
    url: str
    license: str
    files: tuple[str, ...]            # names (basenames) that identify the dataset, inside an archive or loose
    approx_gb: float
    notes: str = ""
    direct_urls: tuple[str, ...] = ()  # public downloads the remote box can fetch itself


AUX: dict[str, AuxDataset] = {
    "dgraph": AuxDataset("dgraph", "DGraph-Fin (Xinye)", "graph", "https://dgraph.xinye.com", "non-commercial, registration",
                         ("dgraphfin.npz",), 0.7, "3.7M users, 4.3M edges, 15.5k fraud, 17 features, timestamps"),
    "elliptic": AuxDataset("elliptic", "Elliptic Bitcoin transactions", "graph", "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
                           "CC BY-NC", ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"), 0.7),
    "yelpchi": AuxDataset("yelpchi", "YelpChi review fraud graph", "graph", "https://github.com/YingtongDou/CARE-GNN", "research use",
                          ("YelpChi.mat",), 0.2),
    "amazon": AuxDataset("amazon", "Amazon review fraud graph", "graph", "https://github.com/YingtongDou/CARE-GNN", "research use",
                         ("Amazon.mat",), 0.2),
    "malurl": AuxDataset("malurl", "Malicious URLs", "text", "https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset", "CC0",
                         ("malicious_phish.csv",), 0.05),
    "fakejob": AuxDataset("fakejob", "Fake job postings (EMSCAD)", "text", "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
                          "CC0", ("fake_job_postings.csv",), 0.05),
    "twitterbot": AuxDataset("twitterbot", "Twitter human / bot accounts", "text", "https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts",
                             "CC0", ("twitter_human_bots_dataset.csv",), 0.02),
    "balabit": AuxDataset("balabit", "Balabit mouse dynamics challenge", "behaviour", "https://github.com/balabit/Mouse-Dynamics-Challenge",
                          "research use", ("public_labels.csv",), 0.3, "10 users, 816 labelled test sessions (405 intruders)"),
    "bbmas": AuxDataset("bbmas", "SU-AIS BB-MAS", "behaviour", "https://ieee-dataport.org/open-access/su-ais-bb-mas-behavioral-biometrics-multi-device-and-multi-activity-data-same-users",
                        "CC BY 4.0", ("BB-MAS_Dataset.zip",), 9.6, "117 users: keystrokes, mouse, swipes, gait on desktop / tablet / phone"),
    "hmog": AuxDataset("hmog", "HMOG touch + motion", "behaviour", "https://hmog-dataset.github.io/hmog/", "research use, request form",
                       ("hmog_dataset.zip",), 6.2, "100 users, 24 sessions each, touch + accelerometer + gyroscope"),
    "brainrun": AuxDataset("brainrun", "BrainRun touch gestures + sensors", "behaviour", "https://zenodo.org/records/2598135", "CC BY 4.0",
                           ("gestures_devices_users_games_data.zip", "sensors_data.zip"), 30.0,
                           direct_urls=("https://zenodo.org/records/2598135/files/gestures_devices_users_games_data.zip?download=1",
                                        "https://zenodo.org/records/2598135/files/sensors_data.zip?download=1")),
    "talkingdata": AuxDataset("talkingdata", "TalkingData AdTracking click fraud", "graph",
                              "https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection", "Kaggle competition rules",
                              ("talkingdata-adtracking-fraud-detection.zip", "train_sample.csv"), 11.0, "optional; 184M clicks with ip/app/device/os"),
}

# loose files or archive members -> key, for everything we know (transaction sources first)
_NAME_TO_KEY: dict[str, str] = {}
for _k, _src in SOURCES.items():
    for _f in _src.files:
        for _alt in _f.split("|"):
            _NAME_TO_KEY[Path(_alt).name.lower()] = _k
    for _z in _src.kaggle_zip_names:
        if _z.lower() != "archive.zip":
            _NAME_TO_KEY[_z.lower()] = _k
for _k, _d in AUX.items():
    for _f in _d.files:
        _NAME_TO_KEY[_f.lower()] = _k
# a few file names worth knowing that the registries list differently
_NAME_TO_KEY.update({"base.csv": "baf", "variant i.csv": "baf", "fraud_data.csv": "fraudecom", "ipaddress_to_country.csv": "fraudecom",
                     "ieee-fraud-detection.zip": "ieee", "train_transaction.csv": "ieee", "hi-small_trans.csv": "amlworld",
                     "sd254_users.csv": "tabformer", "sd254_cards.csv": "tabformer", "creditcard.csv": "ulb",
                     "fraudyelp.zip": "yelpchi", "fraudamazon.zip": "amazon"})


def identify(path: Path) -> str | None:
    """which dataset a raw file or archive belongs to, by its name or the names inside it"""
    path = Path(path)
    key = _NAME_TO_KEY.get(path.name.lower())
    if key:
        return key
    if path.is_dir():
        for p in path.iterdir():
            k = _NAME_TO_KEY.get(p.name.lower())
            if k:
                return k
        return None
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    k = _NAME_TO_KEY.get(Path(n).name.lower())
                    if k:
                        return k
        except Exception:  # noqa: BLE001
            return None
    return None


def scan(raw_root: Path) -> dict[str, list[Path]]:
    """{key: [files]} for everything identifiable under a raw folder (one level of subfolders,
    which is how a Downloads folder tends to look). a subfolder named like a key is taken as is."""
    found: dict[str, list[Path]] = {}
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return found

    def add(key, p):
        found.setdefault(key, []).append(p)

    for p in sorted(raw_root.iterdir()):
        if p.name.startswith(("_", ".")):
            continue
        if p.is_dir() and (p.name in SOURCES or p.name in AUX):
            for f in sorted(p.rglob("*")):
                if f.is_file() and ".git" not in f.parts:
                    add(p.name, f)
            continue
        if p.is_dir():
            # a folder holding one dataset (DGraphFin/) travels whole; a folder holding several
            # (reg-req-data/) is read one entry at a time
            keys = {identify(q) for q in p.iterdir()} - {None}
            if len(keys) == 1:
                only = next(iter(keys))
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        add(only, f)
                continue
            for q in sorted(p.iterdir()):
                k = identify(q)
                if k:
                    if q.is_dir():
                        for f in sorted(q.rglob("*")):
                            if f.is_file():
                                add(k, f)
                    else:
                        add(k, q)
            continue
        k = identify(p)
        if k:
            add(k, p)
    return found


def find_file(raw_root: Path, key: str, name: str) -> Path | None:
    """a raw file of an aux dataset, extracting the folder's archives first if it is only
    inside one (the kaggle zips arrive as 'archive (9).zip')"""
    root = Path(raw_root) / key
    if not root.exists():
        return None
    hits = sorted(root.rglob(name))
    if hits:
        return hits[0]
    for z in sorted(root.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                members = [m for m in zf.namelist() if Path(m).name == name]
                if members:
                    zf.extract(members[0], root)
        except zipfile.BadZipFile:
            continue
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None
