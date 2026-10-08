"""Categories, support queues and global settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

# Phase 1 defaults (synthetic data). Point these at the live artifacts with env vars:
#   ROUTER_MODEL=models/nb_live.joblib ROUTER_KB=data/live/kb_live.json uvicorn src.app:app
MODEL_PATH = Path(os.getenv("ROUTER_MODEL", MODEL_DIR / "nb_router.joblib"))
KB_PATH = Path(os.getenv("ROUTER_KB", DATA_DIR / "kb_articles.json"))
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"

# Phase 2 (live data) locations
LIVE_DIR = DATA_DIR / "live"
RAW_DIR = LIVE_DIR / "raw"            # collectors write *.jsonl here
CACHE_DIR = LIVE_DIR / "cache"        # cached HTTP responses
UNLABELED_PATH = LIVE_DIR / "unlabeled.csv"
LABELED_PATH = LIVE_DIR / "labeled.csv"
LIVE_MODEL_PATH = MODEL_DIR / "nb_live.joblib"
LOG_QUESTIONS = os.getenv("LOG_QUESTIONS", "0") == "1"   # log masked production questions

# ComplementNB probabilities are flatter than usual (chance with 11 classes = 0.09). 0.20 was chosen
# from out-of-fold predictions: ~61% auto-routed at ~98% accuracy (see models/metrics.json).
CONFIDENCE_THRESHOLD = 0.20
TOP_K_ARTICLES = 3

# category -> support queue shown to the agent
CATEGORIES = {
    "refund_status": "Refunds & Amount Owed",
    "efile_pin": "E-File & Rejections",
    "w2_1099_import": "Document Import",
    "state_return": "State Returns",
    "deductions_credits": "Deductions & Credits",
    "self_employed": "Self-Employed & Gig",
    "investments_rental": "Investments, Crypto & Rental",
    "amended_return": "Amended Returns",
    "extension_deadline": "Deadlines & Extensions",
    "account_payment": "Account & Billing",
    "other_human": "Human Agent",
}
LABELS = list(CATEGORIES)
