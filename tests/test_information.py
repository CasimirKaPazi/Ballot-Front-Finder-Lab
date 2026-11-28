"""Tests of tools/information.py (what a method's ballots and count can carry), against answers known
without it. Run from the lab's folder:
python tests/test_information.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import methods  # noqa: E402
from information import analyse  # noqa: E402

assert methods.MAXS == 5


def row(name, C):
    r = analyse(name, C)
    assert isinstance(r, dict) and r["exact"], (name, C, r)
    return r


for C in (3, 4):
    pairs = C * (C - 1) // 2
    fact = 1
    for i in range(2, C + 1):
        fact *= i

    # Plurality: one vertex per candidate, the totals up to a number added to all, and no reverses
    r = row("Plurality", C)
    assert (r["ballots"], r["vertices"], r["rank"], r["symmetric"], r["equality"]) == (C, C, C - 1, False, None), r

    # Approval: every set but the empty and the full one is a vertex; a set and its complement cancel
    r = row("Approval", C)
    assert (r["vertices"], r["rank"], r["symmetric"], r["equality"]) == (2 ** C - 2, C - 1, True, True), r

    # Score: more ballots, the same vertices as Approval (intermediate scores are mixtures)
    s = row("Score", C)
    assert s["ballots"] == 6 ** C and s["counts"] == 6 ** C - 5 ** C, s
    assert (s["vertices"], s["rank"], s["symmetric"], s["equality"]) == (2 ** C - 2, C - 1, True, True), s

    # A method that reads only the margins: one vertex per strict ranking (a truncated ranking is a
    # mixture of the rankings it lies between), one number per pair, and reverses cancel
    for name in ("Minimax", "Schulze", "Ranked pairs"):
        r = row(name, C)
        assert (r["vertices"], r["rank"], r["symmetric"], r["equality"]) == (fact, pairs, True, True), r

    # Winning votes: a tie is no mixture of the two orders, so every weak order but the full tie is a vertex
    r = row("MMWV", C)
    weak = {3: 13, 4: 75}[C]
    assert (r["counts"], r["vertices"], r["equality"]) == (weak, weak - 1, False), r

    # IRV: the first preferences among every subset of at least two candidates, each up to its sum;
    # with three candidates the whole histogram of rankings. A ballot and its reverse do not cancel
    r = row("IRV", C)
    subsets = sum((k - 1) * {3: {2: 3, 3: 1}, 4: {2: 6, 3: 4, 4: 1}}[C][k] for k in range(2, C + 1))
    assert (r["vertices"], r["rank"], r["equality"]) == (fact, subsets, False), r

# Borda with rankings that may be truncated (the unranked share the bottom): truncation adds vertices
r = row("Borda", 3)
assert (r["vertices"], r["rank"], r["equality"]) == (9, 2, True), r
# ... on strict rankings only the rankings, a centrally symmetric polytope
r = row("Inverse-Borda", 3)
assert (r["vertices"], r["rank"], r["symmetric"], r["equality"]) == (6, 2, True, True), r
# Borda-ER gives a tie no points, so a ballot and its reverse do not cancel
assert row("Borda-ER", 3)["equality"] is False

# STAR: the totals and the pairwise preferences
r = row("STAR", 3)
assert (r["rank"], r["symmetric"], r["equality"]) == (5, True, True), r

# What is left out says why
assert isinstance(analyse("Random ballot", 3), str)
assert isinstance(analyse("Condorcet-random-drop", 3), str)

print("ok")
