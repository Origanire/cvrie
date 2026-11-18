from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "supervised"
MODEL = ROOT / "models" / "supervised" / "best_model.joblib"
REPORT = ROOT / "reports" / "supervised" / "metrics.json"
