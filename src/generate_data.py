"""Generate synthetic training questions from templates + realistic noise.

Run:  python -m src.generate_data
Writes data/train_synthetic.csv and data/gold.csv
"""
import csv
import random
import itertools

from .config import DATA_DIR
from .gold import GOLD

random.seed(7)

STATES = ["california", "new york", "texas", "ohio", "florida", "new jersey",
          "illinois", "georgia", "virginia", "pennsylvania", "massachusetts", "michigan"]
SLOTS = {
    "n": ["2", "3", "4", "5", "6", "8", "10"],
    "unit": ["weeks", "days"],
    "kind": ["federal", "state", "tax"],
    "irs": ["accepted", "approved", "processing", "received"],
    "code": ["IND-031-04", "IND-181", "R0000-500-01", "F8962-070", "IND-516"],
    "box": ["1", "2", "12", "14", "16", "17"],
    "state": STATES,
    "state2": STATES,
    "item": ["home office", "medical bills", "student loan interest", "charity donations",
             "property taxes", "daycare costs", "mortgage interest", "gifts", "moving costs",
             "work uniforms", "car registration", "tips income"],
    "platform": ["uber", "lyft", "doordash", "instacart", "etsy", "upwork", "fiverr", "airbnb"],
    "asset": ["bitcoin", "ethereum", "tesla stock", "mutual funds", "etf shares", "crypto", "options"],
    "yrs": ["1", "2", "3"],
    "form": ["1099-nec", "1099-k", "1099-b", "1099-int", "1099-div", "w-2", "1099-r", "1098"],
}

TEMPLATES = {
    "refund_status": [
        "where is my refund", "where is my {kind} refund", "how long does it take to get my {kind} refund",
        "my refund has not arrived after {n} {unit}", "check status of my tax refund",
        "irs shows {irs} but i have no refund yet", "when will my refund be deposited",
        "refund amount is different from the estimate", "why is my refund delayed",
        "how do i pay the amount i owe", "i owe taxes what are the payment options",
        "how to track my {kind} refund", "refund status says {irs} what does that mean",
        "still waiting on refund after {n} {unit}", "how do i see my refund date",
        "will the refund go to my bank account", "balance due how to pay irs",
    ],
    "efile_pin": [
        "my return was rejected by the irs", "efile rejection error {code}", "what is a self select pin",
        "i forgot my prior year agi", "how do i sign my return electronically",
        "irs rejected my return because my dependent was already claimed",
        "do i need to mail anything after e-filing", "was my return accepted by the irs",
        "how do i enter my identity protection pin", "return rejected code {code} how to fix",
        "how to resubmit a rejected return", "where do i find my ip pin",
        "does my spouse need a separate pin", "e-file not going through",
        "what does e-file reject mean", "how long does efile acceptance take",
    ],
    "w2_1099_import": [
        "w2 upload is not working", "how does the ai import work for w-2", "photo of my {form} failed to read",
        "import filled the wrong wages", "can i upload a pdf of my {form}",
        "employer id not detected after scanning", "where do i find box {box} on my w-2",
        "i do not have my w-2 yet", "how do i import last years return",
        "scan of my {form} is blurry what should i do", "import added the same {form} twice",
        "can i take a picture of my {form}", "ai import skipped my {form}",
        "how to manually enter my w-2 instead of upload", "w2 box {box} is missing after import",
        "upload says unsupported file type",
    ],
    "state_return": [
        "how much is a state return", "do i need to file a state return in {state}",
        "i moved from {state} to {state2} during the year how do i file",
        "can i file state only", "state refund status for {state}",
        "which states does olt support", "part year resident state filing",
        "{state} return was rejected", "is state filing an extra cost",
        "how do i add a state return", "do i have to file in {state} if i work remotely",
        "need to file two states {state} and {state2}", "where can i get {state} state tax forms",
        "when will my {state} refund come", "nonresident state return help",
    ],
    "deductions_credits": [
        "can i deduct {item}", "is {item} deductible", "standard deduction or itemized which is better",
        "how to claim the child tax credit", "am i eligible for the earned income credit",
        "how do i claim the new deduction for tips", "where do i enter overtime deduction",
        "how to claim education credit", "do i qualify for the american opportunity credit",
        "car loan interest deduction how does it work", "senior deduction how to claim",
        "how much is the child tax credit per kid", "what deductions am i missing",
        "can i claim my {item} on my return", "which credits do i qualify for",
        "how do i claim dependents for the child credit",
    ],
    "self_employed": [
        "how do i report freelance income", "do i need schedule c for my side business",
        "i received a {form} what do i do", "how to report {form} from {platform}",
        "how is self employment tax calculated", "{platform} driver tax deductions",
        "can i deduct business expenses", "i sell on {platform} how do i file",
        "do i pay estimated taxes as a contractor", "how to deduct mileage for {platform}",
        "sole proprietor tax filing help", "what business expenses can i write off",
        "i work as an independent contractor how do i file", "schedule se explained",
        "home office deduction for my freelance work",
    ],
    "investments_rental": [
        "how are {asset} gains taxed", "where do i enter {asset} sales", "how do i report rental income",
        "rental property depreciation help", "what is cost basis", "i received a 1099-da",
        "capital loss carryover", "how to report dividends", "schedule d for stock sales",
        "schedule e for rental properties", "i sold {asset} at a loss", "short term vs long term capital gains",
        "do i need to report {asset} trades", "rental expenses i can deduct",
        "where do i enter my brokerage 1099-b",
    ],
    "amended_return": [
        "i made a mistake on my filed return", "how do i file form 1040-x", "i forgot to include a {form} after filing",
        "can i amend my return from {yrs} years ago", "check amended return status",
        "i need to change my filing status after filing", "how long does an amended return take",
        "how to correct a return that was already accepted", "how do i amend my return with olt",
        "amend return to add a dependent", "i entered the wrong income and already filed",
        "do i have to amend my state return too", "can i amend electronically",
    ],
    "extension_deadline": [
        "what is the tax filing deadline", "how to file an extension form 4868",
        "i missed the deadline what now", "extension to file or extension to pay",
        "penalty for filing late", "can i still e-file after april 15",
        "how do i get more time to file", "when are taxes due this year",
        "do i owe a penalty if i file an extension", "last day to e-file",
        "file extension online for free", "how long is the extension",
    ],
    "account_payment": [
        "how much does olt cost", "is federal really free", "i was charged twice",
        "how do i pay for my state return", "i forgot my password and cannot sign in",
        "how to delete my account", "i want a refund of my payment", "how do i upgrade to premium",
        "update the email address on my account", "do you have a coupon code",
        "how to cancel premium", "where can i find my old return", "cannot log in to my account",
        "what payment methods do you accept", "unlock my account", "download a copy of my return",
    ],
    "other_human": [
        "i want to talk to a real person", "can someone call me back", "i think someone filed using my ssn",
        "i received an irs audit letter", "i got a cp2000 notice", "i need a tax lawyer",
        "can you file taxes for my s corp", "i want to file a complaint", "this service is terrible",
        "i need legal advice about my divorce and taxes", "hello", "hi", "test", "help",
        "what is the weather", "tell me a joke", "who are you", "asdfgh", "qwerty",
        "can i speak to a manager", "irs levied my bank account", "i need help with a payment plan with the irs",
    ],
}

PREFIX = ["", "", "", "hi ", "hello ", "pls help ", "urgent ", "quick question ", "hey team ", "need help ", "sir "]
SUFFIX = ["", "", "", " thanks", " please help", " ??", " asap", " pls", " thank you", " ?"]


def typo(word: str) -> str:
    if len(word) < 5:
        return word
    i = random.randrange(1, len(word) - 2)
    kind = random.choice(["swap", "drop", "dup"])
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def noisy(text: str) -> str:
    words = text.split()
    words = [typo(w) if random.random() < 0.08 else w for w in words]
    text = " ".join(words)
    if random.random() < 0.4:
        text = text.replace("'", "")
    text = random.choice(PREFIX) + text + random.choice(SUFFIX)
    return text.strip()


def fill(template: str) -> str:
    out = template
    for key, values in SLOTS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, random.choice(values), 1)
    if "{state2}" in template:
        pass
    return out


def build_synthetic(per_template: int = 6):
    rows = set()
    for label, templates in TEMPLATES.items():
        for t in templates:
            for _ in range(per_template):
                rows.add((noisy(fill(t)), label))
    rows = sorted(rows)
    random.shuffle(rows)
    return rows


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rows = build_synthetic()
    with open(DATA_DIR / "train_synthetic.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    gold = [(q, label) for label, qs in GOLD.items() for q in qs]
    with open(DATA_DIR / "gold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(gold)
    print(f"synthetic: {len(rows)} rows | gold: {len(gold)} rows")


if __name__ == "__main__":
    main()
