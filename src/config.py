"""Categories, support queues and global settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

MODEL_PATH = MODEL_DIR / "nb_router.joblib"
KB_PATH = DATA_DIR / "kb_articles.json"
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"

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
