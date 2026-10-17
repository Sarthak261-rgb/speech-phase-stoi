"""Package the submission.

    python make_zip.py          full package: everything except the virtual environment and raw downloads
    python make_zip.py --lite   small package for upload portals: report, notebook, code, tests, tables,
                                clean-signal audio and the scrambled clean files
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LITE = "--lite" in sys.argv
OUT = ROOT.parent / ("speech-phase-stoi_submission_lite.zip" if LITE else "speech-phase-stoi_submission.zip")
INCLUDE = ["README.md", "requirements.txt", "pyproject.toml", "Speech_Phase_STOI.ipynb",
           "report/Speech_Phase_STOI_Report.pdf", "phaselab.py", "experiments.py", "run_all.py", "recover.py",
           "make_report.py", "report_text.py", "build_notebook.py", "make_zip.py", "outputs/results.json"]
if LITE:
    TREES = ["tests", "data/speech", "outputs/tables", "outputs/audio/clean"]
    EXTRA = sorted((ROOT / "outputs" / "challenge").glob("clean__*.wav"))
else:
    TREES = ["tests", "data/speech", "data/noise", "outputs"]
    EXTRA = []

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    files = [ROOT / f for f in INCLUDE] + EXTRA
    for t in TREES:
        files += [p for p in sorted((ROOT / t).rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    for p in dict.fromkeys(files):
        z.write(p, f"speech-phase-stoi/{p.relative_to(ROOT).as_posix()}")
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
