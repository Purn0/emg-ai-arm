from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../emg_ai_arm
DATA_DIR = PROJECT_ROOT / "data"
WINDOWS_DIR = DATA_DIR / "windows"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

for p in [DATA_DIR, WINDOWS_DIR, RAW_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
