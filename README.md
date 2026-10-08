# Tax Support Question Router (Naive Bayes)

Routes a customer's tax question to a **support queue** and the **top 3 knowledge-base articles**.
Low-confidence questions go to a human instead of being auto-routed.

Idea source: OLT.com / Pistus.com (support questions on refunds, e-file, W-2 import, state returns, deductions).

**Phase 2 (live data instead of synthetic) is in `README_PHASE2.md`.**

## How it works
```
question -> clean + mask PII -> HashingVectorizer (1-2 grams) -> Naive Bayes -> class probabilities
         -> confidence >= 0.20 ?  yes: queue + KB articles (TF-IDF similarity inside the predicted category)
                                  no : "needs a person"
agent correction -> POST /feedback -> partial_fit (online learning) + saved to data/feedback.jsonl
```
11 categories: refund_status, efile_pin, w2_1099_import, state_return, deductions_credits, self_employed,
investments_rental, amended_return, extension_deadline, account_payment, other_human.

## Run it
The full write-up with charts is in `report.ipynb`. It is self-contained (no imports from `src/`) and can be run from anywhere.

```bash
pip install -r requirements.txt
python -m src.generate_data     # synthetic training data + gold test questions
python -m src.train             # trains, compares models, writes models/ + metrics.json
uvicorn src.app:app --reload    # UI at http://localhost:8000  |  API docs at /docs
pytest -q
docker build -t support-router . && docker run -p 8000:8000 support-router
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /classify` `{"question": "..."}` | category, confidence, queue, decision, alternatives, top_articles |
| `POST /feedback` `{"question": "...", "correct_category": "..."}` | online learning from an agent correction |
| `GET /metrics` | requests, auto-routed vs human, corrections, p50/p95 latency |
| `GET /categories`, `GET /health` | housekeeping |

## Data (what is real and what is not)
- `data/kb_articles.json`: 30 KB/help pages. Titles and URLs come from links on the olt.com homepage;
  the matching keywords are **my own short summaries**, not scraped article text. The rental entry links to the
  KB index because I had no direct rental article URL.
- `data/train_synthetic.csv`: ~1,000 questions generated from templates + typos/filler (`src/generate_data.py`).
- `src/gold.py`: 110 hand-written questions in different wording, used for testing (and a small part of training).
- No real customer data. PII (SSN, email, phone) is masked before the model and before anything is logged.

## Results (honest version)
Test = 44 held-out hand-written questions. Training = synthetic + the other 66 hand-written ones.

| Model | Accuracy | Macro-F1 | Latency |
|---|---|---|---|
| MultinomialNB | 0.73 | 0.73 | ~0.02 ms |
| **ComplementNB (served)** | **0.89** | **0.89** | ~0.02 ms |
| Logistic Regression | 0.86 | 0.87 | ~0.04 ms |

Caveats:
- 44 test questions is small. In repeated cross-validation over all 110, the spread was about ±0.08, so
  NB vs Logistic Regression differences are **within noise**.
- ComplementNB was chosen using cross-validation over all 110 questions, after I had seen the single-split
  result, so treat 0.89 as optimistic. A larger real test set is the first thing to add.
- Confidence threshold 0.20 (from out-of-fold predictions): about 61% auto-routed at about 98% accuracy.
  ComplementNB probabilities are flat, so this is a ranking score more than a true probability.
- Online learning works (one correction raised a test question's confidence from 0.15 to 0.20), but
  corrections are only as good as the agents' labels.

## Roadmap
1. Scrape the real KB article text (check robots.txt and terms first, or ask OLT) and replace my keyword summaries.
2. Add real public phrasing (r/tax, r/taxhelp via the official API) and label 300+ questions as a proper test set.
3. Add Tanglish / non-English phrasing and a seasonal test (January vs April topics).
4. Compare against a small transformer (DistilBERT) for the NB-vs-transformer cost/accuracy story.
5. Deploy (Render / HF Spaces) and add a live metrics chart.
