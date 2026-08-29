"""Work out a folder structure from SEM file names ("Smart sort").

The library panel is a virtual tree, so tidying it is purely a naming problem:
read each file's name, decide where it belongs, hand the caller a plan.

Two levels, never more (user rule, 2026-07-30 — the first version nested
Pattern/Janus/Büyükler and that was too much):

    UÖ - 05 /
        Alt
        Karışık
        Üst
        Patterned          <- janus, stripe, undercooled, as produced… all here

The classifier is NOT a fixed keyword list. Known words (the regions and the
pattern classes) are recognised because they must map onto those particular
folders; everything else groups by WHAT THE NAME ACTUALLY SAYS. Strip the sample
("TD - 13") and the photo's index number off the end, and whatever is left is the
group — so a colleague whose files read "TD - 13 x 5 cm 2.jpeg" gets

    TD - 13 /
        x 5 cm
        x 7 cm

without anyone teaching this module about centimetres. A leftover that only one
file has is not worth a folder, so that file stays directly under its sample.

Nothing may depend on the ORDER of the parts, the separator, capitalisation, or
Turkish diacritics being typed: Karışık / Karisik / KARIŞIK all read the same.
"""
from __future__ import annotations

import collections
import os
import re
import unicodedata

# The classes the user works with, plus the process states they keep alongside
# them: every one of these lands in the single "Patterned" folder.
PATTERN_WORDS = {
    "janus": "Janus", "stripe": "Stripe", "striped": "Stripe",
    "cizgili": "Stripe", "lamellar": "Lamellar", "lameller": "Lamellar",
    "lamella": "Lamellar", "composite": "Composite", "kompozit": "Composite",
    "undercooled": "Undercooled", "under cooled": "Undercooled",
    "asiri sogutulmus": "Undercooled", "as produced": "As Produced",
    "asproduced": "As Produced", "uretildigi gibi": "As Produced",
}
# Sampling position / mixture — one folder each, directly under the sample.
REGION_WORDS = {
    "alt": "Alt", "asagi": "Alt",
    "ust": "Üst", "uest": "Üst", "yukari": "Üst",
    "karisik": "Karışık", "mixed": "Karışık",
}
PATTERNED = "Patterned"
UNSORTED = "Unsorted"


def fold(s: str) -> str:
    """Lower-case, strip diacritics and punctuation → a comparable token string.

    Turkish needs care: İ/I/ı/i all collapse to "i", and ş/ç/ğ/ö/ü lose their
    marks, so "KARIŞIK", "Karisik" and "karışık" become the same string.
    """
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("ı", "i").replace("İ", "i").replace("I", "i")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[|_/\\\-–—.,()\[\]*]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _has(hay: str, needle: str) -> bool:
    """Whole-token search that tolerates a glued-on index.

    "alt" matches "05 alt 3" AND "UÖ - 13 Alt1" (a real name in the collection —
    no space before the number), but not "alternatif". Letters are barred on both
    sides while a trailing digit is allowed; a LEADING digit still blocks the
    match, so "6 saat" cannot fire inside "16 saat".
    """
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z])", hay) is not None


def _known(folded: str, table: dict) -> str | None:
    """Canonical name of the first table entry present (longest alias first)."""
    for key in sorted(table, key=len, reverse=True):
        if _has(folded, key):
            return table[key]
    return None


# "UÖ - 05", "TD-13", "UÖ 5" — a short letter code and the sample number, kept in
# the author's own spelling so the folder reads "UÖ - 05" and not "uo".
#
# It must be ANCHORED to the start of the name or to a separator. An unanchored
# "letters then number" pattern looks identical to a class word followed by the
# photo's index — "UÖ - 03 Karışık 1" would then be read as sample "ık - 01",
# which is exactly what the first version did.
_PREFIXED = re.compile(
    r"(?:^|[|/\\_])\s*([^\W\d_]{1,4})\s*(?:[-–—]\s*|\s+)(\d{1,3})(?![\d])")
_LEADING_NUM = re.compile(r"^\s*(\d{1,3})(?![\d])")


def _nfc(s: str) -> str:
    """macOS hands out file names in NFD, where "Ö" is O + a combining mark and
    every letter-matching regex falls apart. Everything here starts from NFC."""
    return unicodedata.normalize("NFC", s)


def sample_of(stem: str):
    """(prefix or None, "05" or None) — the experiment this photo belongs to.

    Three ways it is written, tried in order: a letter code with the number
    ("UÖ - 05", "TD - 13"); a bare number at the start ("05 | Büyükler | …"); or,
    when the parts have been reordered, a number standing ALONE between
    separators ("Janus 12 | 05 | Büyükler" — "12" is glued to a word, so it is
    that photo's index and must not win).
    """
    stem = _nfc(stem)
    m = _PREFIXED.search(stem)
    if m:
        return m.group(1), m.group(2).zfill(2)
    m = _LEADING_NUM.match(stem)
    if m:
        return None, m.group(1).zfill(2)
    for part in re.split(r"[|/\\_]+", stem):
        p = part.strip()
        if re.fullmatch(r"\d{1,3}", p):
            return None, p.zfill(2)
    return None, None


def _strip_sample(stem: str) -> str:
    """The name with the sample marker removed (the rest is what groups it)."""
    stem = _nfc(stem)
    out = _PREFIXED.sub(" ", stem, count=1)
    if out == stem:
        out = _LEADING_NUM.sub(" ", stem, count=1)
    return out


def _tidy(s: str) -> str:
    """Separators → spaces, trimmed. Keeps the author's own capitalisation."""
    s = re.sub(r"[|/\\_]+", " ", s)
    s = re.sub(r"[-–—]+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" -–—.,*#~+")


def _drop_index(s: str) -> str:
    """Remove the photo's index: the trailing number, glued on or not.

    "x 5 cm 2" -> "x 5 cm";  "Alt1" -> "Alt";  "5 cm" is NOT touched, because
    its number carries a unit after it.
    """
    return re.sub(r"(?<=[^\W\d_])\d{1,3}\s*$|(?<!\S)\d{1,3}\s*$", "", s).strip()


def group_of(stem: str):
    """(display name, matching key, is_known) for this photo, or (None, None, False).

    Known words win, so the regions and the pattern classes always land in their
    own folders; anything else is grouped by the leftover text itself. The flag
    says which of the two happened — the fixed folders exist even for a single
    photo, a discovered one has to earn its place (see plan()).
    """
    stem = _nfc(stem)
    f = fold(stem)
    region = _known(f, REGION_WORDS)
    if region:
        return region, fold(region), True
    if _known(f, PATTERN_WORDS):
        return PATTERNED, fold(PATTERNED), True
    rest = _drop_index(_tidy(_strip_sample(stem)))
    if not re.search(r"[^\W_]", rest):        # nothing but punctuation left
        return None, None, False
    return rest, fold(rest), False


def plan(paths):
    """[(path, [folders…])] for every file, plus the ones nothing was read from.

    Sample folders are labelled with the letter code the collection uses for that
    number, even for files that omit it ("05 | Büyükler | Janus 9" files under
    "UÖ - 05" like their siblings). A leftover group only becomes a folder when
    at least two photos share it — one-offs stay directly under their sample.
    """
    parsed = {}
    prefixes = collections.defaultdict(collections.Counter)   # number -> codes
    keys = collections.Counter()                              # (sample, key)
    spelling = collections.defaultdict(collections.Counter)   # key -> names
    for p in paths:
        stem = _nfc(os.path.splitext(os.path.basename(p))[0])
        prefix, num = sample_of(stem)
        name, key, known = group_of(stem)
        parsed[p] = (prefix, num, name, key, known)
        if num and prefix:
            prefixes[num][prefix] += 1
        if num and key:
            keys[(num, key)] += 1
            spelling[key][name] += 1

    # A number whose own files never spell the code out still belongs to the
    # collection ("05 | Büyükler | …" is UÖ - 05). Borrowing the code is only
    # safe when the whole library uses ONE — with UÖ and TD side by side there is
    # nothing to infer, so those samples stay bare numbers.
    all_codes = {c for cnt in prefixes.values() for c in cnt}
    lone_code = next(iter(all_codes)) if len(all_codes) == 1 else None

    out, unsorted_ = [], []
    for p, (prefix, num, name, key, known) in parsed.items():
        if not num:
            unsorted_.append(p)
            out.append((p, [UNSORTED]))
            continue
        code = prefixes[num].most_common(1)[0][0] if prefixes[num] else lone_code
        sample = f"{code} - {num}" if code else num
        folders = [sample]
        if key and (known or keys[(num, key)] > 1):
            folders.append(spelling[key].most_common(1)[0][0])
        out.append((p, folders))
    return out, unsorted_


def tree_of(plan_rows):
    """Nested {folder: {...}} + a "" key holding the files, for the preview."""
    root = {}
    for p, folders in plan_rows:
        node = root
        for f in folders:
            node = node.setdefault(f, {})
        node.setdefault("", []).append(p)
    return root


def outline(node, depth=0, lines=None):
    """The plan as indented text (what the confirmation dialog shows)."""
    lines = [] if lines is None else lines
    for name in sorted(k for k in node if k != ""):
        sub = node[name]
        lines.append("    " * depth + f"{name}  ({_count(sub)})")
        outline(sub, depth + 1, lines)
    return lines


def _count(node):
    n = len(node.get("", ()))
    for k, v in node.items():
        if k != "":
            n += _count(v)
    return n
