"""Tokenizing for retrieval: PII-safe cleaning, stop-word removal, Porter stemming."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer

from ..model import clean

_STEM = PorterStemmer()
STOP = set("""a an the of to in on for and or is are was were be been am i my me we you your it its this that these those
with at by from as do does did how what when where why which who can could should would will if then than so about into
over after before any some there here have has had get got""".split())


@lru_cache(maxsize=50000)
def stem(w: str) -> str:
    return _STEM.stem(w)


def tokens(text: str, keep_stop: bool = False) -> list:
    """lower-case, mask PII, stem. 'w-2' also yields 'w2'."""
    out = []
    for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", clean(text or "")):
        if not keep_stop and w in STOP:
            continue
        out.append(stem(w))
        if "-" in w:
            out.append(stem(w.replace("-", "")))
    return out
