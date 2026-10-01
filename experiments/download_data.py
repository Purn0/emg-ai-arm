"""
Download the UCI "EMG data for Gestures" dataset (CC BY 4.0) into
data/uci_emg/ and copy subject 05's two sessions to data/raw/, where the
app's replay source and calibration look for them.

    python -m experiments.download_data

Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S. (2018).
EMG Data for Gestures [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5ZP5C
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile

from experiments.data import DEFAULT_DATA, REPO_ROOT, session_paths

URL = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
RAW_DIR = REPO_ROOT / "emg_ai_arm" / "data" / "raw"


def main():
    target = DEFAULT_DATA.parent
    if not DEFAULT_DATA.exists():
        print(f"Downloading {URL} ...")
        with urllib.request.urlopen(URL) as response:
            payload = response.read()
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as outer:
            outer.extractall(target)
        # The UCI archive may wrap the data folder in a second zip.
        for inner in target.glob("*.zip"):
            with zipfile.ZipFile(inner) as z:
                z.extractall(target)
            inner.unlink()

    paths = session_paths(DEFAULT_DATA)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for session in (1, 2):
        dest = RAW_DIR / f"subject05_session{session}.txt"
        shutil.copyfile(paths[(5, session)], dest)
    print(f"{len(paths)} session files in {DEFAULT_DATA}")
    print(f"subject 05 sessions copied to {RAW_DIR}")


if __name__ == "__main__":
    main()
