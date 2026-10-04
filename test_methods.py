"""Hand-built checks. Run: python test_methods.py"""
import numpy as np
from methods import METHODS
from metrics import election_row


def profile(groups):
    return np.array([u for n, u in groups for _ in range(n)], dtype=float)


def check(name, got, want):
    ok = np.allclose(got, want, atol=1e-9)
    print(f"{'ok ' if ok else 'FAIL'} {name}: {np.round(got, 3)}")
    assert ok


# 4 R>G>B>K, 3 B>G>R>K, 2 G>R>B>K: G is the Condorcet winner, R has most first choices
u = profile([(4, [0.50, 0.30, 0.15, 0.05]), (3, [0.15, 0.30, 0.50, 0.05]), (2, [0.30, 0.50, 0.15, 0.05])])
check("Plurality", METHODS["Plurality"].honest(u), [1, 0, 0, 0])
check("IRV (K out, G out, R wins)", METHODS["IRV"].honest(u), [1, 0, 0, 0])
check("Copeland", METHODS["Copeland"].honest(u), [0, 1, 0, 0])
check("BTR-score", METHODS["BTR-score"].honest(u), [0, 1, 0, 0])
check("MARS", METHODS["MARS"].honest(u), [0, 1, 0, 0])
check("MARA (3-5)", METHODS["MARA"].honest(u), [0, 1, 0, 0])
check("Random ballot", METHODS["Random ballot"].honest(u), [4 / 9, 2 / 9, 3 / 9, 0])

# New methods on the same 4-3-2 profile
check("Random pair runoff", METHODS["Random pair runoff"].honest(u), [36 / 72, 30 / 72, 6 / 72, 0])
check("Top-two runoff (R vs B)", METHODS["Top-two runoff"].honest(u), [1, 0, 0, 0])
check("Approval runoff (G vs R)", METHODS["Approval runoff"].honest(u), [0, 1, 0, 0])
check("BTR-Borda", METHODS["BTR-Borda"].honest(u), [0, 1, 0, 0])
check("Minimax", METHODS["Minimax"].honest(u), [0, 1, 0, 0])
check("Baldwin (K out, B out, G beats R)", METHODS["Baldwin"].honest(u), [0, 1, 0, 0])
check("MJ (R and G median 3, R wins on removal)", METHODS["MJ"].honest(u), [1, 0, 0, 0])
check("MARA-Top", METHODS["MARA-Top"].honest(u), [1, 0, 0, 0])

# Methods added later, same 4-3-2 profile (G is the Condorcet winner)
check("Anti-plurality (everyone vetoes K: R, G, B tie)", METHODS["Anti-plurality"].honest(u), [1 / 3, 1 / 3, 1 / 3, 0])
check("Condorcet-random-drop", METHODS["Condorcet-random-drop"].honest(u), [0, 1, 0, 0])
check("SICT", METHODS["SICT"].honest(u), [0, 1, 0, 0])
check("MDDA (R and B dominated by G)", METHODS["MDDA"].honest(u), [0, 1, 0, 0])
check("BTR-IRV (K out vs G, B out vs G, G beats R)", METHODS["BTR-IRV"].honest(u), [0, 1, 0, 0])
check("Benham (Condorcet winner at once)", METHODS["Benham"].honest(u), [0, 1, 0, 0])

# TRASH: plurality winner R, anti-plurality all veto K -> R, G, B tie for fewest lasts.
# Runoffs R vs G (G preferred 5-4, so R wins), R vs B (R preferred 6-3, so B wins), R vs R (R).
check("TRASH", METHODS["TRASH"].honest(u), [2 / 3, 0, 1 / 3, 0])
# MARA-Bottom: every non-zero rating is approval
check("MARA-Bottom", METHODS["MARA-Bottom"].honest(u), [0, 1, 0, 0])

# Exact three-way tie: each method shares probability equally among R, G, B
t = profile([(3, [0.55, 0.17, 0.15, 0.13]), (3, [0.15, 0.55, 0.17, 0.13]), (3, [0.17, 0.15, 0.55, 0.13])])
for m in ["Plurality", "IRV", "Copeland", "Score", "STAR", "BTR-score", "MARS", "MARA", "MARA-Top", "MJ",
          "Top-two runoff", "Approval runoff", "BTR-Borda", "Minimax", "Baldwin", "Borda",
          "Anti-plurality", "SICT", "MDDA", "BTR-IRV", "Benham"]:
    check(f"{m} three-way tie", METHODS[m].honest(t), [1 / 3, 1 / 3, 1 / 3, 0])

# Baseline lottery from the example: utilities 100, 50, 25, 25 -> 50%, 25%, 12.5%, 12.5%
one = np.array([[100, 50, 25, 25]], dtype=float)
r = election_row(one, np.array([1, 0, 0, 0.0]))
check("tau=1 baseline utility", np.array([r["U_tau1"]]), [100 * .5 + 50 * .25 + 25 * .125 * 2])

for m, fn in METHODS.items():
    p = fn.honest(u)
    assert abs(p.sum() - 1) < 1e-9, m
print("all probabilities sum to 1")

# Condorcet-random-drop on a cycle: R>G>B, G>B>R, B>R>G (3 each), K last for all.
# Dropping voters resolves the cycle; the result should be close to symmetric and never K.
cyc = profile([(3, [0.5, 0.3, 0.15, 0.05]), (3, [0.05, 0.5, 0.3, 0.15]), (3, [0.3, 0.05, 0.5, 0.15])])
p = METHODS["Condorcet-random-drop"].honest(cyc)
print("Condorcet-random-drop on a cycle:", np.round(p, 3))
assert abs(p.sum() - 1) < 1e-9 and p[3] == 0 and p[:3].min() > 0.15

# SICT: equal-first ranking by a compromising faction keeps the compromise unbeaten
w = np.array([[3, 3, 1, 0]] * 4 + [[1, 2, 3, 0]] * 3 + [[2, 3, 1, 0]] * 2, dtype=float)   # 4 voters rank R=G top
print("SICT with R and G ranked equal first by 4 voters:", np.round(METHODS["SICT"].tally(w), 3))

from methods import select
assert select(["star", "SICT"]) == ["STAR", "SICT"]
assert "Plurality" not in select(None, ["plurality"]) and len(select()) == len(METHODS)
print("method selection ok")
