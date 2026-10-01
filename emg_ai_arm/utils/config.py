from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../emg_ai_arm
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # subject 05 sessions for the replay (experiments.download_data)
MODELS_DIR = PROJECT_ROOT / "models"  # trained models (experiments.train_models)

for p in (RAW_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)
