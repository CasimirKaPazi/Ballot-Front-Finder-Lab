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

# More methods on the same 4-3-2 profile (G is the Condorcet winner)
check("Anti-plurality (everyone vetoes K: R, G, B tie)", METHODS["Anti-plurality"].honest(u), [1 / 3, 1 / 3, 1 / 3, 0])
check("Condorcet-random-drop", METHODS["Condorcet-random-drop"].honest(u), [0, 1, 0, 0])
check("SICT", METHODS["SICT"].honest(u), [0, 1, 0, 0])
check("MDDA (R and B dominated by G)", METHODS["MDDA"].honest(u), [0, 1, 0, 0])
check("BTR-IRV (K out vs G, B out vs G, G beats R)", METHODS["BTR-IRV"].honest(u), [0, 1, 0, 0])
check("Benham (Condorcet winner at once)", METHODS["Benham"].honest(u), [0, 1, 0, 0])


# More methods on the same 4-3-2 profile
check("Ranked pairs", METHODS["Ranked pairs"].honest(u), [0, 1, 0, 0])
check("Schulze", METHODS["Schulze"].honest(u), [0, 1, 0, 0])
check("Approval cond. runoff (G approved by all 9)", METHODS["Approval cond. runoff"].honest(u), [0, 1, 0, 0])
check("TRASH (runner-ups; B vs R/G/B: B always wins)", METHODS["TRASH"].honest(u), [0, 0, 1, 0])
check("Score+ (only K eliminated, G wins after renormalising)", METHODS["Score+"].honest(u), [0, 1, 0, 0])
p = METHODS["Reciprocal score"].honest(u)   # printed for inspection (no hand-computed expectation)
print("Reciprocal score on 4-3-2:", np.round(p, 3)); assert abs(p.sum() - 1) < 1e-9

# On the 4-3-2 profile: G is the Condorcet winner; R has most first choices and the best Dowdall
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
check("SICT with R and G ranked equal first by 4 voters (G stays the winner)", METHODS["SICT"].tally(w), [0, 1, 0, 0])

from methods import select
assert select(["star", "SICT"]) == ["SICT", "STAR"]            # in the order of METHODS
assert "Plurality" not in select(None, ["plurality"]) and len(select()) == len(METHODS)
from methods import GROUPS
assert all(m in METHODS for ms in GROUPS.values() for m in ms)
assert "Schulze" in select(["Condorcet", "STAR"]) and "Benham" in select(["condorcet-hybrid"]) and "STAR" in select(["condorcet", "star"])
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

# Score range (set_max_score): sincere scores stretch to 0..max; ARA approves above the middle;
# feature sums stay exact; learning runs. Restored to the default afterwards.
import methods as _m
from strategy import mutate
for S in (10, 100):
    _m.set_max_score(S)
    B = METHODS["Score"].sincere(u)
    assert B.min() == 0 and B.max() == S and np.all(B == np.rint(B)), S
    up = _m.FEATURES["upper"][0](np.array([[S / 2, S / 2 + 1, S]]))
    assert up.tolist() == [[False, True, True]], S
    for name, m in METHODS.items():
        if m.ballot == "score" and m.full is None:
            Bs = m.sincere(u); F = m.features(Bs); B2 = Bs.copy(); B2[:3] = Bs[-1]
            S2 = F.sum(axis=0) + (m.features(B2[:3]) - F[:3]).sum(axis=0)
            assert np.allclose(m.decide_many(S2[None], Bs.shape[1])[0], m.tally(B2)), (name, S)
            assert abs(m.tally(Bs).sum() - 1) < 1e-9, (name, S)
    r = np.random.default_rng(5)
    vals = {mutate(np.full(4, S // 2 + 1.0), "score", r)[0] for _ in range(4000)}
    assert vals <= set(range(S + 1)) and {0, S} <= vals, S
_m.set_max_score(5)
check("Score+ chicken example, default range again", METHODS["Score+"].tally(sp), [1, 0, 0, 0])
print("score range 0..10 and 0..100 ok")

# Strategic nomination (nomination.py)
from nomination import nominate, clone_setup, utilities, to_original, moves, Field
from scenario import Scenario, electorate_positions


def spatial(vpos, cpos, width=1.5, seed=0):
    vp, cp = np.array(vpos, float)[:, None], np.array(cpos, float)[:, None]
    u = np.exp(-np.abs(vp - cp.T) / width)
    return (u, *clone_setup(vp, cp, width, np.random.default_rng(seed)))


# Vote splitting: A (0.0) and B (0.1) share the left, C (1.0) has the right. Plurality elects C;
# B (or A) withdraws, and the other one of the two wins.
u, uc, cu = spatial([-0.1, -0.05, 0.0, 0.08, 0.12, 0.2, 0.9, 1.0, 1.0, 1.1], [0.0, 0.1, 1.0])
P0 = METHODS["Plurality"].honest(u)
f, st = nominate(METHODS["Plurality"], u, uc, cu, np.random.default_rng(1), exits=1)
P1 = to_original(METHODS["Plurality"].honest(utilities(f.entries(), u, uc)), f.entries(), 3)
check("Plurality before nomination", P0, [0, 0, 1])
assert P1[2] == 0 and st["nom_exits"] == 1 and st["nom_settled"] == 1, (P1, st)
print("nomination: vote splitting resolved by an exit", np.round(P1, 3))

# A clone is ranked next to its original by every voter
u, uc, cu = spatial(np.linspace(-1, 2, 21), [0.0, 0.5, 1.0])
g = Field(3); g.clones[1] = 0
uf = utilities(g.entries(), u, uc)
assert np.all(np.abs(np.argsort(-uf, axis=1).argsort(axis=1)[:, 1] - np.argsort(-uf, axis=1).argsort(axis=1)[:, 3]) == 1)
print("nomination: clones ranked next to their originals ok")

# Settled fields are equilibria: no candidate has a strictly improving move
sc = Scenario(n_voters=31, n_cands=5)
for name in ["Plurality", "Borda", "IRV", "Minimax", "Score"]:
    m = METHODS[name]
    for e in range(4):
        u, vp, cp = electorate_positions(sc, np.random.default_rng([7, e]))
        uc, cu = clone_setup(vp, cp, sc.width, np.random.default_rng([7, e, 1]))
        f, st = nominate(m, u, uc, cu, np.random.default_rng(e), exits=1, clones=2)
        if not st["nom_settled"]:
            continue
        def val(i, g):
            ents = g.entries(); P = m.honest(utilities(ents, u, uc))
            return to_original(P, ents, 5) @ cu[i]
        for i in range(5):
            assert all(val(i, g) <= val(i, f) + 1e-12 for g in moves(f, i, 1, 2)), (name, e, i)
        assert f.inn.sum() >= 2 and len(f.clones) <= 2
print("nomination: settled fields are equilibria ok")

# Learning works on fields of 2 candidates (strategic exits can leave only two)
from strategy import learn
for name, m in METHODS.items():
    learn(m, np.random.default_rng(1).random((15, 2)), np.random.default_rng(2), 2, 2, 5, None)
print("learning on 2 candidates ok")

# Poll rounds (strategy.respond)
from strategy import respond, default_poll_size
assert (default_poll_size(100), default_poll_size(99), default_poll_size(200)) == (51, 49, 101)
P_ = METHODS["Plurality"]
# 4 voters A>C>B, 3 voters B>A>C, 5 voters C>A>B: C wins honestly. With an exact poll (poll size = all
# voters), only B's voters can do better: they compromise on A, and A wins. A's voters gain nothing
# by moving (B is their last choice), C's voters are winning.
u = profile([(4, [0.6, 0.0, 0.3, 0.0]), (3, [0.4, 0.6, 0.0, 0.0]), (5, [0.3, 0.0, 0.6, 0.0])])
u = u[:, :3] + 0.01                                     # three candidates
B, st, _ = respond(P_, u, np.random.default_rng(0), rounds=1, samples=3, poll_size=len(u), search=20)
check("Plurality after one poll round (B's voters compromise)", P_.tally(B), [1, 0, 0])
B0, st0, s0 = respond(P_, u, np.random.default_rng(0), rounds=0)
assert np.array_equal(B0, P_.sincere(u)) and st0 and s0 == 1.0
mask = np.zeros(len(u), bool); mask[:4] = True          # only A's voters strategic: nothing changes
B, st, _ = respond(P_, u, np.random.default_rng(0), rounds=2, samples=3, poll_size=len(u), search=20, strategic=mask)
assert np.array_equal(B, P_.sincere(u)) and st
print("poll rounds ok")

# Leader rule emerges (poll rounds): A leads B clearly; voters who approve both drop the leader they
# like less, on both sides, although the gap is large; nobody approves both front-runners any more.
Ap = METHODS["Approval"]
u = profile([(10, [0.9, 0.7, 0.0]), (6, [0.7, 0.9, 0.0]), (4, [0.9, 0.0, 0.2]), (3, [0.0, 0.2, 0.9])]) + 0.01
Bs = Ap.sincere(u)
assert list(Bs.sum(axis=0)) == [20, 16, 3], Bs.sum(axis=0)
B, _, _ = respond(Ap, u, np.random.default_rng(1), rounds=1, samples=30, poll_size=0, search=30)
assert (B[:10, 1] == 0).all() and (B[10:16, 0] == 0).all(), B[:16]
print("leader rule on both sides ok")

# Aggressive vs kind strategic voters (poll rounds): A leads by far under Borda; burying A cannot
# change the result. Aggressive voters do it anyway (it cannot hurt), kind ones stay sincere.
Bo = METHODS["Borda"]
u = profile([(30, [0.9, 0.5, 0.1]), (4, [0.5, 0.9, 0.1]), (3, [0.1, 0.5, 0.9])]) + 0.01
for aggr, expect_buried in ((True, True), (False, False)):
    B, _, _ = respond(Bo, u, np.random.default_rng(2), rounds=1, aggressive=np.full(len(u), aggr))
    buried = bool((B[30:34, 0] < B[30:34, 2]).all())       # B's voters put A below C
    assert buried == expect_buried, (aggr, B[30:34])
    assert Bo.tally(B)[0] == 1
print("aggressive vs kind ok")

# Poll rounds on 2 candidates (strategic exits can leave only two): every ballot type runs
u2 = r.random((21, 2))
for name, m in METHODS.items():
    B, _, _ = respond(m, u2, np.random.default_rng(1), rounds=2)
    assert abs(m.tally(B).sum() - 1) < 1e-9, name
print("poll rounds on 2 candidates ok")

# Several round counts from one run (report=...) equal separate runs with the same random stream
ur = r.random((31, 4))
for name in ["Approval", "IRV", "Borda", "STAR", "TTR"]:
    m = METHODS[name]
    rep = respond(m, ur, np.random.default_rng(4), report=[1, 2, 3])
    for k, (B, st, sn) in zip([1, 2, 3], rep):
        B1, st1, sn1 = respond(m, ur, np.random.default_rng(4), rounds=k)
        assert np.array_equal(B, B1) and st == st1 and abs(sn - sn1) < 1e-12, (name, k)
print("reused rounds = separate runs ok")

u = profile([(6, [0.9, 0.3, 0.1]), (5, [0.3, 0.9, 0.1])]) + 0.01
B, st, _ = respond(METHODS["Plurality"], u, np.random.default_rng(0), rounds=1)
assert np.array_equal(B, METHODS["Plurality"].sincere(u)) and st
print("settled ok")

# Equal-ranking variants equal their strict versions on sincere (strict) ballots
r = np.random.default_rng(11)
pairs = [("Borda", "Borda-ER"), ("Baldwin", "Baldwin-ER"), ("Minimax", "Minimax-ER"),
         ("Schulze", "Schulze-ER"), ("Ranked Robin", "Ranked Robin-SR"), ("Minimax", "MMWV")]
for _ in range(100):
    uu = r.random((int(r.integers(5, 40)), int(r.integers(3, 7))))
    for a_, b_ in pairs:
        assert np.allclose(METHODS[a_].honest(uu), METHODS[b_].honest(uu)), (a_, b_)
print("ER variants = strict versions on strict ballots ok")

# IC-MARS: 3 voters A5 B4, 2 voters A3 B5, 4 voters A5 B5 (C 0). A beats B 3:2, but the 4 voters
# rating both at the top could reverse that, so the margin counts 0 and B's higher score (42 > 41)
# wins the final; MARS gives A the margin (1 - 1/5 > 0) and elects A.
sc = np.array([[5, 4, 0]] * 3 + [[3, 5, 0]] * 2 + [[5, 5, 0]] * 4, dtype=float)
check("MARS on the tied-at-top example", METHODS["MARS"].tally(sc), [1, 0, 0])
check("IC-MARS on the tied-at-top example", METHODS["IC-MARS"].tally(sc), [0, 1, 0])
# MARS-T scales the margin by the share of ballots scoring A and B differently (5/9): 5/9 - 1/5 > 0 -> A.
# With the 2 B voters rating A 2 instead of 3 (A 39, B 42, a gap of 3 points): MARS 1 - 3/5 > 0 -> A,
# MARS-T 5/9 - 3/5 < 0 -> B.
check("MARS-T on the tied-at-top example", METHODS["MARS-T"].tally(sc), [1, 0, 0])
sc2 = np.array([[5, 4, 0]] * 3 + [[2, 5, 0]] * 2 + [[5, 5, 0]] * 4, dtype=float)
check("MARS, score gap 3", METHODS["MARS"].tally(sc2), [1, 0, 0])
check("MARS-T, score gap 3", METHODS["MARS-T"].tally(sc2), [0, 1, 0])
# MMWV+Score and MMWVxScore: an undefeated candidate with the highest score wins
sc2 = np.array([[5, 3, 0]] * 5 + [[2, 5, 0]] * 3 + [[4, 0, 5]] * 1, dtype=float)
check("MMWV+Score", METHODS["MMWV+Score"].tally(sc2), [1, 0, 0])
check("MMWVxScore", METHODS["MMWVxScore"].tally(sc2), [1, 0, 0])

# Kemeny-Young = brute force over all rankings (winner set = heads of optimal rankings)
from itertools import permutations as _perms
for _ in range(60):
    uu = r.random((int(r.integers(5, 30)), int(r.integers(3, 6))))
    B = METHODS["Borda"].sincere(uu)
    gt = (B[:, :, None] > B[:, None, :]).sum(axis=0)
    C = gt.shape[0]
    sc = {o: sum(gt[o[i], o[j]] for i in range(C) for j in range(i + 1, C)) for o in _perms(range(C))}
    top = max(sc.values())
    heads = sorted({o[0] for o, v in sc.items() if v == top})
    P = METHODS["Kemeny-Young"].tally(B)
    assert sorted(np.flatnonzero(P > 0)) == heads, (P, heads)
print("Kemeny-Young = brute force ok")

# Coombs: 4 A>B>C, 3 C>B>A, 2 B>C>A. No first-choice majority (4 of 9); A has most last places
# (5) and goes; then B vs C: B first on 4+2 = 6 > 4.5, B wins. IRV instead drops B (fewest first
# choices); B's voters go to C, and C beats A 5:4.
co = profile([(4, [0.6, 0.3, 0.1]), (3, [0.1, 0.3, 0.6]), (2, [0.1, 0.6, 0.3])])
check("Coombs (A most last places, then B)", METHODS["Coombs"].honest(co), [0, 1, 0])
check("IRV on the same profile (B out, then C beats A)", METHODS["IRV"].honest(co), [0, 0, 1])

# IRV-3 equals IRV when 3 rankings cover everyone; with more candidates exhausted ballots stop counting
for _ in range(50):
    uu = r.random((int(r.integers(5, 30)), 3))
    assert np.allclose(METHODS["IRV-3"].honest(uu), METHODS["IRV"].honest(uu))
t3 = METHODS["IRV-3"].sincere(r.random((4, 6)))
assert (np.sort(t3, axis=1)[:, ::-1][:, :4] == [3, 2, 1, 0]).all()
# 5 candidates: 3 voters rank A>B>C, 3 voters rank D>E>A, 4 voters rank C>D>B (others unranked).
# IRV-3: firsts A3 D3 C4 (B, E 0 -> out one at a time); then A3 D3 C4: A and D tie for last, ...
ex = np.array([[3, 2, 1, 0, 0]] * 3 + [[1, 0, 0, 3, 2]] * 3 + [[0, 1, 3, 2, 0]] * 4, dtype=float)
P = METHODS["IRV-3"].tally(ex)
assert abs(P.sum() - 1) < 1e-9 and P[4] == 0 and P[1] == 0
print("IRV-3 ok", np.round(P, 3))

# Methods that agree on every field must agree with nomination on too: the candidates' order of moves
# is the same for every method (it once was keyed by method name, so such pairs differed honestly).
from run import one_election
rows = []
for e in range(6):
    rows += one_election((Scenario(n_voters=31, n_cands=5), 3, e, [1], 5, 10, 0.5, 5, 1, 2, 0, 2.0, 2, 0, 0.5,
                          ["Borda", "Borda-ER", "IRV", "ER-IRV"], False, "honest", None))
hon = {}
for row in rows:
    if row["ballots"] == "sincere":
        hon.setdefault(row["method"], []).append(row["U_method"])
assert np.allclose(hon["Borda"], hon["Borda-ER"]) and np.allclose(hon["IRV"], hon["ER-IRV"]), hon
print("ER pairs: same honest result with nomination ok")

# Two-round elections: the runoff is a separate vote, read from the fixed sincere ranking, so
# strategy changes only the round-1 marks. Honestly TTR = top two by first choices, then the majority
# runoff on the same rankings (the contingent vote).
from methods import Method, d_top_two
contingent = Method("rank", d_top_two, ("first", "gt"))
for _ in range(100):
    uu = r.random((int(r.integers(5, 40)), int(r.integers(3, 7))))
    assert np.allclose(METHODS["TTR"].honest(uu), contingent.honest(uu))
# 4 A>B>C, 3 B>A>C, 2 C>B>A. Honestly A 4, B 3, C 2: A and B go on, B wins the runoff 5:4.
tr = profile([(4, [0.9, 0.5, 0.1]), (3, [0.5, 0.9, 0.1]), (2, [0.1, 0.5, 0.9])])
Bt = METHODS["TTR"].sincere(tr)
check("TTR honest (A, B in the runoff, B wins 5:4)", METHODS["TTR"].tally(Bt), [0, 1, 0])
Bt[7:, :] = Bt[7:, :] % 3 + 3 * np.eye(3)[0]          # the C voters mark A: A 6 of 9 wins outright,
check("TTR: round-1 majority wins outright", METHODS["TTR"].tally(Bt), [1, 0, 0])   # though B beats A
# Marks decide the finalists, the sincere rankings the runoff: 4 A>B>C, 4 C>B>A, 3 B>A>C; two A voters
# mark B (A 2, B 5, C 4, no majority of 11): B and C go on, B wins the runoff 7:4.
tr2 = profile([(4, [0.9, 0.5, 0.1]), (4, [0.1, 0.5, 0.9]), (3, [0.5, 0.9, 0.1])])
B2 = METHODS["TTR"].sincere(tr2)
check("TTR honest (A, C in the runoff, A wins 7:4)", METHODS["TTR"].tally(B2), [1, 0, 0])
B2[:2, :] = B2[:2, :] % 3 + 3 * np.eye(3)[1]
check("TTR: finalists by marks, runoff by rankings", METHODS["TTR"].tally(B2), [0, 1, 0])
from strategy import mutate as _mut
bb = METHODS["Approval runoff"].sincere(r.random((1, 5)))[0]
for _ in range(50):
    nb = _mut(bb, "app_run", r)
    assert np.array_equal(nb % 5, bb % 5) and 0 < (nb >= 5).sum() < 5
    nb = _mut(METHODS["TTR"].sincere(r.random((1, 5)))[0], "fav_run", r)
    assert (nb >= 5).sum() == 1
print("two-round elections: strategy changes only round-1 marks ok")

# Participation (participation.py). Feature subtraction equals tallying without the absent ballots;
# Score and Approval never fail participation with sincere ballots; IRV and Minimax sometimes do.
import participation as part
ident = lambda P: P
def brute(m, B, u):
    P0 = m.tally(B)
    for g in part._groups(B, u):
        for k in sorted({1, (len(g) + 1) // 2, len(g)}):
            keep = np.ones(len(B), bool); keep[g[:k]] = False
            if ((m.tally(B[keep]) - P0) @ u[g[:k]].T).sum() > part.TOL:
                return 1
    return 0
r = np.random.default_rng(5)
fails = {k: 0 for k in ("Score", "Approval", "IRV", "Minimax", "TTR", "Borda")}
for _ in range(150):
    uu = r.random((int(r.integers(7, 25)), 4)) ** 2
    for name in fails:
        m = METHODS[name]
        B = m.sincere(uu)
        x = part.no_show(m, B, uu, uu, ident)
        assert x["noshow"] == brute(m, B, uu), name
        fails[name] += x["noshow"]
assert fails["Score"] == 0 and fails["Approval"] == 0 and fails["IRV"] > 0, fails
print("participation check ok; no-show cases in 150 small elections:", fails)

# Every method is in exactly one group (asserted in methods.py as well)
from methods import GROUPS, group_of
assert all(group_of(m) != "other" for m in METHODS) and sum(map(len, GROUPS.values())) == len(METHODS)
print("groups: every method in exactly one ok")

# Inverse-* methods: the sincere ballot expresses reversed preferences; reversing it back gives the
# original method. On the 4-3-2 profile K is last for everyone (Condorcet loser, least liked by all).
from strategy import sincerity
u = profile([(4, [0.50, 0.30, 0.15, 0.05]), (3, [0.15, 0.30, 0.50, 0.05]), (2, [0.30, 0.50, 0.15, 0.05])])
for inv, orig in [("Inverse-Plurality", "Plurality"), ("Inverse-Approval", "Approval"), ("Inverse-Minimax", "Minimax"),
                  ("Inverse-IRV", "IRV"), ("Inverse-ER-IRV", "ER-IRV"), ("Inverse-Borda", "Borda"),
                  ("Inverse-Score", "Score")]:
    mi, mo = METHODS[inv], METHODS[orig]
    assert np.allclose(mi.tally(mo.sincere(u)), mo.honest(u))          # original's sincere ballots, same count
    assert sincerity(mi.sincere(u), mi.pref(u)).min() == 1               # honest = sincere for its own ballot
check("Inverse-Minimax elects the Condorcet loser", METHODS["Inverse-Minimax"].honest(u), [0, 0, 0, 1])
check("Inverse-Plurality elects the most often least liked", METHODS["Inverse-Plurality"].honest(u), [0, 0, 0, 1])
# Inverse-TTR: marking the favourite in round 1 gives TTR (the runoff reads the reversed ranking back)
Bt = METHODS["Inverse-TTR"].sincere(u)
assert sincerity(Bt, METHODS["Inverse-TTR"].pref(u)).min() == 1
Bt = Bt % 4 + 4 * METHODS["Plurality"].sincere(u)                         # marks: favourites
assert np.allclose(METHODS["Inverse-TTR"].tally(Bt), METHODS["TTR"].honest(u))
for _ in range(50):
    uu = r.random((int(r.integers(5, 30)), int(r.integers(3, 6))))
    C = uu.shape[1]
    Bt = METHODS["Inverse-TTR"].sincere(uu) % C + C * METHODS["Plurality"].sincere(uu)
    assert np.allclose(METHODS["Inverse-TTR"].tally(Bt), METHODS["TTR"].honest(uu))
print("inverse methods ok")

# Candidate dispersion and shift change only the candidates; voters are the same draws
from scenario import electorate_positions as _ep
a = _ep(Scenario(n_voters=31, n_cands=5), np.random.default_rng(9))
b = _ep(Scenario(n_voters=31, n_cands=5, cand_dispersion=0.5, cand_shift=1.0), np.random.default_rng(9))
assert np.array_equal(a[1], b[1])
c0 = a[1].mean(axis=0)
want = c0 + 0.5 * (a[2] - c0); want[:, 0] += 1.0 * a[1][:, 0].std()
assert np.allclose(b[2], want)
print("candidate dispersion and shift ok")

# DSC and DAC. Benham's example: 46 A (B, C tied below), 44 B>C, 10 C (A, B tied below).
# DSC: {A} is the strongest solid coalition (46) -> A. DAC: {A,C} acquiesced by 56, then {B,C} by 54 -> C.
bx = np.array([[2, 0, 0]] * 46 + [[0, 2, 1]] * 44 + [[0, 0, 2]] * 10, dtype=float)
check("DSC on Benham's example", METHODS["DSC"].tally(bx), [1, 0, 0])
check("DAC on Benham's example", METHODS["DAC"].tally(bx), [0, 0, 1])
for _ in range(100):                          # strict rankings: DSC = DAC (solid = acquiescing sets)
    uu = r.random((int(r.integers(5, 30)), int(r.integers(3, 6))))
    assert np.allclose(METHODS["DSC"].honest(uu), METHODS["DAC"].honest(uu))
    assert abs(METHODS["DSC"].honest(uu).sum() - 1) < 1e-9

# STLR: A totals 3x5+2x1 = 17, B 3x4+2x5 = 22, C 0: A and B go on. Levelled: the first 3 voters give
# A 5, B 4; the other 2 give A 1, B 5: A 15 + 2 = 17, B 12 + 10 = 22 -> B. With 0..5 scores 2 and 1 on
# one ballot the levelled scores are 5 and 2.5 (electowiki's example).
st = np.array([[5, 4, 0]] * 3 + [[1, 5, 0]] * 2, dtype=float)
check("STLR", METHODS["STLR"].tally(st), [0, 1, 0])
L = FEATURES_ = __import__("methods").FEATURES["levelled"][0](np.array([[2.0, 1.0, 0.0]])).reshape(3, 3)
assert abs(L[0, 1] - 5) < 1e-9 and abs(L[1, 0] - 2.5) < 1e-9 and L[2, 0] == 0

# INS (iterated normalized score): 2 voters A5 B4 C0, 2 voters C5 B4 A0, 1 voter B5 A0 C0. Totals A10 B13
# C10: A and C tie for last (split). Without A: the C voters' ballots stretch to C5 B0... see below.
ins = np.array([[5, 4, 0]] * 2 + [[0, 4, 5]] * 2 + [[0, 5, 0]], dtype=float)
p = METHODS["INS"].tally(ins)
# A out: ballots over {B,C}: A-voters B5 C0, C-voters B0 C5, B-voter B5 C0 -> B 15, C 10 -> B.
# C out: over {A,B}: A-voters A5 B0, C-voters A0 B5, B-voter A0 B5 -> B 15 -> B.
check("INS", p, [0, 1, 0])

# COWPEA Lottery: exact probabilities equal a simulation of the lottery
cw_ = np.array([[1, 1, 0]] * 6 + [[0, 1, 1]] * 3 + [[1, 0, 0]] * 1, dtype=float)
P = METHODS["COWPEA Lottery"].tally(cw_)
rs = np.random.default_rng(1); wins = np.zeros(3)
for _ in range(20000):
    S = set(np.flatnonzero(cw_[rs.integers(10)]))
    for _ in range(200):
        if len(S) == 1:
            break
        T = set(np.flatnonzero(cw_[rs.integers(10)]))
        if S & T and S & T != S:
            S &= T
    for c in S:
        wins[c] += 1 / len(S)
assert np.allclose(P, wins / wins.sum(), atol=0.015), (P, wins / wins.sum())
print("COWPEA Lottery = simulation ok", np.round(P, 3))

# MAMPO: 4 voters A>B (A, B approved), 3 voters B>C, 2 voters C (truncated). Approvals A 4, B 7, C 5 of
# 9: B and C have a majority. Largest opposition: B 4 (A over B), C 7 (B over C: 4 + 3) -> B.
# Without a majority approval the most approved wins.
mp = np.array([[2, 1, 0]] * 4 + [[0, 2, 1]] * 3 + [[0, 0, 1]] * 2, dtype=float)
check("MAMPO", METHODS["MAMPO"].tally(mp), [0, 1, 0])
mp2 = np.array([[1, 0, 0]] * 4 + [[0, 1, 0]] * 3 + [[0, 0, 1]] * 2, dtype=float)
check("MAMPO without majority approval (most approved)", METHODS["MAMPO"].tally(mp2), [1, 0, 0])


# Soft Minimax: with clipping at 1 point on integer 0..5 scores, a soft margin is the majority margin,
# so Soft Minimax is Minimax on the score ballots; Soft Minimax-40 counts 1-point differences as half
from methods import d_minimax, Method
mm_on_scores = Method("score", d_minimax, ("gt",))
for _ in range(100):
    uu = r.random((int(r.integers(5, 30)), int(r.integers(3, 6))))
    assert np.allclose(METHODS["Soft Minimax"].honest(uu), mm_on_scores.honest(uu))
sm = np.array([[5, 4, 0]] * 3 + [[3, 5, 0]] * 2, dtype=float)
# A over B: 3 voters by 1 point (half each) = 1.5, 2 voters by 2 (full) the other way: soft margin -0.5 -> B
check("Soft Minimax-40 (1-point differences count half)", METHODS["Soft Minimax-40"].tally(sm), [0, 1, 0])
check("Soft Minimax (= Minimax on scores: A beats B 3:2)", METHODS["Soft Minimax"].tally(sm), [1, 0, 0])

# Soft Minimax-L and -D. 3 voters A5 B4 C0, 2 voters A3 B5 C0: A averages 4.2, B 4.4, clip 4.3 for A-B
# under -L: A over B: 3 x 1/4.3 - 2 x 2/4.3 < 0 -> B (like Score). Under -D the clip is 1 + 0.2 = 1.2:
# A over B 3 x 1/1.2 - 2 x 1 > 0 -> A (close averages: nearly a majority vote).
check("Soft Minimax-L (clip = mean average score of the pair)", METHODS["Soft Minimax-L"].tally(sm), [0, 1, 0])
check("Soft Minimax-D (clip = 1 + difference of the averages)", METHODS["Soft Minimax-D"].tally(sm), [1, 0, 0])

# Geometric median: on 2 candidates with ballots (5,0) x3 and (0,5) x2 it is the majority's ballot (3 > 2,
# the median sits at a data point), unlike the mean (Score would also pick A here); exaggeration test:
# a 1-ballot minority moving further away does not move the median.
gm = np.array([[5, 0, 2]] * 3 + [[0, 5, 2]] * 2, dtype=float)
check("Geometric median (majority ballot)", METHODS["Geometric median"].tally(gm), [1, 0, 0])
x1 = METHODS["Geometric median"].full(np.array([[3, 2, 0]] * 4 + [[2, 3, 0]] * 3 + [[0, 3, 5]], dtype=float))
x2 = METHODS["Geometric median"].full(np.array([[3, 2, 0]] * 4 + [[2, 3, 0]] * 3 + [[0, 5, 5]], dtype=float))
assert np.allclose(x1, x2)
print("geometric median ok")
