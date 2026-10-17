"""Shared fixtures, plus a small hook that writes every test's outcome to
outputs/tables/test_results.csv so the report can list the test cases."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phaselab as pl  # noqa: E402

_RESULTS = []


@pytest.fixture(scope="session", autouse=True)
def prepared_data():
    """On a fresh clone the 16 kHz files do not exist yet; create them from data/raw_*."""
    needed = [ROOT / "data" / "speech" / "original.wav"] + [ROOT / "data" / "noise" / f"{n}_16k.wav"
                                                             for n in ("white", "babble", "machinegun")]
    if not all(f.exists() for f in needed):
        import experiments
        experiments.prepare_data()


@pytest.fixture(scope="session")
def speech(prepared_data):
    """First 4 s of the clean recording (enough for STOI, quick to process)."""
    x = pl.load_wav(str(ROOT / "data" / "speech" / "original.wav"))
    return x[: 4 * pl.FS]


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
        doc = (item.function.__doc__ or "").strip().split("\n")[0]
        _RESULTS.append({"id": item.nodeid.split("::", 1)[-1], "file": Path(item.fspath).name,
                         "description": doc, "outcome": rep.outcome, "seconds": round(rep.duration, 3)})


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    out = ROOT / "outputs" / "tables" / "test_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_RESULTS[0]))
        w.writeheader()
        w.writerows(_RESULTS)
