"""Hand-built checks. Run: python test_methods.py"""
import numpy as np
from methods import METHODS
from metrics import election_row, worst_margins


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
check("ARA (3-5)", METHODS["ARA"].honest(u), [0, 1, 0, 0])

# New methods on the same 4-3-2 profile
check("Random pair runoff", METHODS["Random pair runoff"].honest(u), [36 / 72, 30 / 72, 6 / 72, 0])
check("TTR (R vs B)", METHODS["TTR"].honest(u), [1, 0, 0, 0])
check("Approval runoff (G vs R)", METHODS["Approval runoff"].honest(u), [0, 1, 0, 0])
check("BTR-Borda", METHODS["BTR-Borda"].honest(u), [0, 1, 0, 0])
check("Minimax", METHODS["Minimax"].honest(u), [0, 1, 0, 0])
check("Baldwin (K out, B out, G beats R)", METHODS["Baldwin"].honest(u), [0, 1, 0, 0])
check("MJ (R and G median 3, R wins on removal)", METHODS["MJ"].honest(u), [1, 0, 0, 0])
check("ARA-Top", METHODS["ARA-Top"].honest(u), [1, 0, 0, 0])

# Methods added later, same 4-3-2 profile (G is the Condorcet winner)
check("Anti-plurality (everyone vetoes K: R, G, B tie)", METHODS["Anti-plurality"].honest(u), [1 / 3, 1 / 3, 1 / 3, 0])
check("Condorcet-random-drop", METHODS["Condorcet-random-drop"].honest(u), [0, 1, 0, 0])
check("SICT", METHODS["SICT"].honest(u), [0, 1, 0, 0])
check("MDDA (R and B dominated by G)", METHODS["MDDA"].honest(u), [0, 1, 0, 0])
check("BTR-IRV (K out vs G, B out vs G, G beats R)", METHODS["BTR-IRV"].honest(u), [0, 1, 0, 0])
check("Benham (Condorcet winner at once)", METHODS["Benham"].honest(u), [0, 1, 0, 0])


# Methods added in the third batch, same 4-3-2 profile
check("Ranked pairs", METHODS["Ranked pairs"].honest(u), [0, 1, 0, 0])
check("Schulze", METHODS["Schulze"].honest(u), [0, 1, 0, 0])
check("Approval cond. runoff (G approved by all 9)", METHODS["Approval cond. runoff"].honest(u), [0, 1, 0, 0])
check("TRASH (runner-ups; B vs R/G/B: B always wins)", METHODS["TRASH"].honest(u), [0, 0, 1, 0])
check("Score+ (only K eliminated, G wins after renormalising)", METHODS["Score+"].honest(u), [0, 1, 0, 0])
p = METHODS["Reciprocal score"].honest(u)
print("Reciprocal score on 4-3-2:", np.round(p, 3)); assert abs(p.sum() - 1) < 1e-9

# Batch 4 on the 4-3-2 profile: G is the Condorcet winner; R has most first choices and the best Dowdall
# score (6 against 5.5); in the supplementary vote the G voters' second choice R beats B 6:3. With three
# grades (good/OK/bad) G has most approvals (MCA) and, in 3-2-1, the fewest 'bad' ratings, then beats R 5:4.
for m, want in [("Black", [0, 1, 0, 0]), ("Ranked Robin", [0, 1, 0, 0]), ("Nanson", [0, 1, 0, 0]),
                ("321", [0, 1, 0, 0]), ("ER-IRV", [1, 0, 0, 0]), ("Dowdall", [1, 0, 0, 0]),
                ("Smith//Score", [0, 1, 0, 0]), ("Supplementary vote", [1, 0, 0, 0]), ("MCA", [0, 1, 0, 0]),
                ("MMPO", [0, 1, 0, 0]), ("MMWV", [0, 1, 0, 0])]:
    check(m, METHODS[m].honest(u), want)

# MMPOWV on the 4-3-2 profile: G has no defeat and largest opposition 4 of 9: 1 x 5/9; R is beaten 5:4:
# 4/9 x 4/9 -> G.
check("MMPOWV", METHODS["MMPOWV"].honest(u), [0, 1, 0, 0])
# With an equal ranking: 2 voters A > B > C, 2 voters B > A = C, 1 voter C > A > B. A beats B 3:2 and C 2:1
# (two voters tie A = C), B beats C 4:1. MMPOWV: A has no defeat, largest opposition 2 (B over A) -> 1 x 3/5;
# B: defeat 3, opposition 3 -> 2/5 x 2/5; C: defeat 4, opposition 4 -> 1/5 x 1/5. A wins.
mw = np.array([[2, 1, 0]] * 2 + [[1, 2, 1]] * 2 + [[1, 0, 2]], dtype=float)
check("MMPOWV with equal rankings", METHODS["MMPOWV"].tally(mw), [1, 0, 0])

# ER-IRV with an equal ranking: 3 voters rank A = B first, 2 voters C > B > A. First votes A 3, B 3, C 2;
# C is out, its voters go to B: B wins 5:3.
er = np.array([[2, 2, 0]] * 3 + [[0, 1, 2]] * 2, dtype=float)
check("ER-IRV with equal top ranks", METHODS["ER-IRV"].tally(er), [0, 1, 0])

# Ranked Robin on the cycle R>G>B>K (4), G>B>K>R (3), B>R>K>G (2): R, G, B each win 2 pairs; margins among
# them R +3 -1 = 2, G -3 +5 = 2, B -5 +1 = -4, so R and G remain; against all candidates K adds R +3, G +5: G.
cy4 = profile([(4, [0.5, 0.3, 0.15, 0.05]), (3, [0.05, 0.5, 0.3, 0.15]), (2, [0.3, 0.05, 0.5, 0.15])])
check("Ranked Robin tiebreak on a cycle", METHODS["Ranked Robin"].honest(cy4), [0, 1, 0, 0])

# Exact three-way tie: each method shares probability equally among R, G, B
t = profile([(3, [0.55, 0.17, 0.15, 0.13]), (3, [0.15, 0.55, 0.17, 0.13]), (3, [0.17, 0.15, 0.55, 0.13])])
for m in ["Plurality", "IRV", "Copeland", "Score", "STAR", "BTR-score", "MARS", "ARA", "ARA-Top", "MJ",
          "TTR", "Approval runoff", "BTR-Borda", "Minimax", "Baldwin", "Borda",
          "Anti-plurality", "SICT", "MDDA", "BTR-IRV", "Benham",
          "Ranked pairs", "Schulze", "Approval cond. runoff", "Reciprocal score", "Score+",
          "Black", "Ranked Robin", "Nanson", "321", "ER-IRV", "Dowdall", "Smith//Score", "Supplementary vote",
          "MCA", "MMPO", "MMWV", "MMPOWV"]:
    check(f"{m} three-way tie", METHODS[m].honest(t), [1 / 3, 1 / 3, 1 / 3, 0])

# Reference points: on the 4-3-2 profile the random ballot elects R, B, G with 4/9, 3/9, 2/9.
# Worst margins (share of 9): R -1/9 (loses to G 4:5), G +1/9 (beats R 5:4 and B 6:3), B -3/9, K -1.
r = election_row(u, np.array([0, 1, 0, 0.0]), worst_margins(u))
check("random-ballot utility", np.array([r["U_rb"]]), [(4 * u[:, 0].sum() + 2 * u[:, 1].sum() + 3 * u[:, 2].sum()) / 9])
check("WM of G, best WM, random-ballot WM", np.array([r["WM"], r["WM_best"], r["WM_rb"]]),
      [1 / 9, 1 / 9, (4 * -1 / 9 + 2 * 1 / 9 + 3 * -3 / 9) / 9])

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
from methods import GROUPS
assert all(m in METHODS for ms in GROUPS.values() for m in ms)
assert "Benham" in select(["Condorcet", "STAR"]) and "STAR" in select(["condorcet", "star"])
assert "TRASH" not in select(None, ["references"]) and "IRV" in select(["plurality-based"])
print("method selection ok")

# Ranked pairs and Schulze on a cycle: R>G>B (4), G>B>R (3), B>R>G (2). Margins R-G 3, G-B 5, B-R 1.
# Ranked pairs locks G>B, then R>G, skips B>R: R wins. Schulze agrees here.
cy = profile([(4, [0.5, 0.3, 0.15, 0.05]), (3, [0.05, 0.5, 0.3, 0.15]), (2, [0.3, 0.05, 0.5, 0.15])])
check("Ranked pairs on a cycle", METHODS["Ranked pairs"].honest(cy), [1, 0, 0, 0])
check("Schulze on a cycle", METHODS["Schulze"].honest(cy), [1, 0, 0, 0])

# Score+ protects a shared compromise: two factions rate each other high; a strong outsider only
# narrowly ahead in total is not a robust winner over the faction that shares support with it.
sp = np.array([[5, 4, 0, 0]] * 4 + [[4, 5, 0, 0]] * 3 + [[0, 0, 5, 0]] * 2, dtype=float)
print("Score+ chicken example:", np.round(METHODS["Score+"].tally(sp), 3))

# Conditional top-two runoff (no runoff when a candidate has more than half the first choices) is the
# same method as TTR here: a majority of first choices also wins every runoff, because the
# runoff reuses the same rankings. So it is not a separate method. Checked on random rankings:
r = np.random.default_rng(3)
for _ in range(500):
    n, C = int(r.integers(3, 30)), int(r.integers(3, 7))
    B = np.array([r.permutation(C) for _ in range(n)], dtype=float)
    first = np.bincount(B.argmax(axis=1), minlength=C)
    if first.max() > n / 2:
        assert METHODS["TTR"].tally(B)[first.argmax()] == 1
print("conditional top-two runoff = top-two runoff ok")

# Feature sums give the same winners as the ballots they come from, also when built incrementally
for name, m in METHODS.items():
    if m.full is None:
        B = m.sincere(u)
        F = m.features(B)
        S = F.sum(axis=0)
        B2 = B.copy(); B2[:3] = B[-1]
        S2 = S + (m.features(B2[:3]) - F[:3]).sum(axis=0)
        assert np.allclose(m.decide_many(S2[None], B.shape[1])[0], m.tally(B2)), name
print("incremental feature sums ok")

# Winner's worst margin: a 70:29 Condorcet winner has +41 points, the loser -41;
# in a 2:1 cycle every candidate's worst margin is -33 points
from metrics import worst_margins
clear = profile([(70, [0.9, 0.5, 0.1]), (29, [0.1, 0.5, 0.9])])
check("worst margins, clear winner", worst_margins(clear), [41 / 99, -41 / 99, -41 / 99])
cyc3 = profile([(33, [0.9, 0.5, 0.1]), (33, [0.1, 0.9, 0.5]), (33, [0.5, 0.1, 0.9])])
check("worst margins, 2:1 cycle", worst_margins(cyc3), [-1 / 3, -1 / 3, -1 / 3])
