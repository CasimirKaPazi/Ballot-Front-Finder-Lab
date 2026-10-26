"""Hand-built checks. Run: python test_methods.py"""
import numpy as np
from methods import METHODS
from metrics import election_row, worst_margins
import methods as _methods_mod
_methods_mod.APPROVAL_DEFAULT = 0.0   # the hand-built checks below use the threshold at the voter's mean (draw 0)


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
# MAV: grades R 5,5,5,5,1,1,1,3,3; G 3x4, 3x3, 5x2. First majority at grade 3 (R 6, G 9 of 9);
# of those, the most votes at grade 4 or more: R 4, G 2 (Bucklin would elect G, the larger majority)
check("MAV (majorities at grade 3: R has more votes above)", METHODS["MAV"].honest(u), [1, 0, 0, 0])
check("MAV (a majority at the top grade: the largest wins)",
      METHODS["MAV"].honest(profile([(3, [1.0, 0.0, 0.5]), (2, [0.0, 1.0, 0.5])])), [1, 0, 0])
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
for m, want in [("Black", [0, 1, 0, 0]), ("Copeland//Margins-ER", [0, 1, 0, 0]), ("Nanson", [0, 1, 0, 0]),
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

# Copeland//Margins on the cycle R>G>B>K (4), G>B>K>R (3), B>R>K>G (2): R, G, B each win 2 pairs; margins among
# them R +3 -1 = 2, G -3 +5 = 2, B -5 +1 = -4, so R and G remain; against all candidates K adds R +3, G +5: G.
cy4 = profile([(4, [0.5, 0.3, 0.15, 0.05]), (3, [0.05, 0.5, 0.3, 0.15]), (2, [0.3, 0.05, 0.5, 0.15])])
check("Copeland//Margins tiebreak on a cycle", METHODS["Copeland//Margins-ER"].honest(cy4), [0, 1, 0, 0])

# Exact three-way tie: each method shares probability equally among R, G, B
t = profile([(3, [0.55, 0.17, 0.15, 0.13]), (3, [0.15, 0.55, 0.17, 0.13]), (3, [0.17, 0.15, 0.55, 0.13])])
for m in ["Plurality", "IRV", "Copeland", "Score", "STAR", "BTR-score", "MARS", "ARA", "ARA-Top", "MJ",
          "TTR", "Approval runoff", "BTR-Borda", "Minimax", "Baldwin", "Borda",
          "Anti-plurality", "SICT", "MDDA", "BTR-IRV", "Benham",
          "Ranked pairs", "Schulze", "Approval cond. runoff", "Reciprocal score", "Score+",
          "Black", "Copeland//Margins-ER", "Nanson", "321", "ER-IRV", "Dowdall", "Smith//Score", "Supplementary vote",
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
assert "Schulze" in select(["Condorcet", "STAR"]) and "Benham" in select(["condorcet-based"]) and "STAR" in select(["condorcet", "star"])
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

# Adaptive voters (per round, by their stake in the leading race): personal thresholds above 1 never
# act (= only the always-strategic voters), below 0 always act once there is a runner-up (= all of them
# strategic); thresholds in between give a share between; a voter indifferent between the leaders never acts
_ua = np.random.default_rng(12).random((41, 4))
_d = np.random.default_rng(13).random(41)
_s, _band = _d < 0.2, (_d >= 0.2) & (_d < 0.8)
for name in ["Approval", "Score", "IRV", "Minimax"]:
    m = METHODS[name]
    run_ = lambda strat, q=None, act=None: respond(m, _ua, np.random.default_rng(5), report=[1, 2, 3],
                                                   strategic=strat, adaptive=q, activated=act)
    _act = {}
    for (B, *_), (B0, *_) in zip(run_(_s, np.where(_band, 1.01, np.nan), _act), run_(_s)):
        assert np.array_equal(B, B0), name
    assert all(v == 0 for v in _act.values()), (name, _act)
    _act = {}
    for (B, *_), (B1, *_) in zip(run_(_s, np.where(_band, -0.01, np.nan), _act), run_(_s | _band)):
        assert np.array_equal(B, B1), name
    assert _act[0] == 0 and all(_act[k] == 1 for k in (1, 2, 3)), (name, _act)
    _act = {}
    run_(_s, np.where(_band, (_d - 0.2) / 0.6, np.nan), _act)
    assert all(0 < _act[k] < 1 for k in (1, 2, 3)), (name, _act)
_ui = np.array([[1.0, 0.0, 0.5]] * 10 + [[0.0, 1.0, 0.5]] * 9 + [[0.5, 0.5, 1.0]] * 6)   # leaders 0 and 1
_act = {}
respond(METHODS["Plurality"], _ui, np.random.default_rng(3), rounds=1, strategic=np.zeros(25, bool),
        adaptive=np.zeros(25), activated=_act)
assert abs(_act[1] - 19 / 25) < 1e-12, _act        # the last six are indifferent between the leaders
print("adaptive voters ok")

# Manipulation test: 4 A>B>C, 3 B>C>A, 2 C>B>A. Plurality elects A; the 5 voters preferring B (or C) to A
# win by voting for it, and A's 4 supporters cannot stop that. Minimax elects the Condorcet winner B: A's
# supporters burying B make C the Condorcet winner (worse for them), C's supporters cannot move B
from manipulation import manipulation
_um = profile([(4, [1.0, 0.5, 0.0]), (3, [0.0, 1.0, 0.5]), (2, [0.0, 0.5, 1.0])])
_r = manipulation(METHODS["Plurality"], METHODS["Plurality"].sincere(_um), _um, _um, lambda P: P)
assert _r["manip1"] == 1 and _r["manip2"] == 1 and _r["U_manip1"] < _um.sum(axis=0)[0] + 1e-9, _r
_r = manipulation(METHODS["Minimax"], METHODS["Minimax"].sincere(_um), _um, _um, lambda P: P)
assert _r["manip1"] == 0 and _r["manip2"] == 0 and abs(_r["U_manip1"] - _um.sum(axis=0)[1]) < 1e-9, _r
# the defence tries every exaggeration of its ballot type: with a sincere Condorcet winner, no Condorcet
# method is manipulable two-sided, whether on strict rankings (Minimax) or rankings with ties (MMWV, whose
# first exaggeration is a weak defence; using only that one let MMWV fail in about a fifth of elections)
import models as _mdm
from scenario import Scenario as _Sc
_rg = np.random.default_rng(5)
_n_cw = 0
for _k in range(25):
    _ue = _mdm.draw_electorate(_Sc(n_voters=31, n_cands=4, model="noise"), _rg).u
    _wins = ((_ue[:, :, None] > _ue[:, None, :]).sum(0) > len(_ue) / 2)
    if not (_wins.sum(1) == 3).any():
        continue
    _n_cw += 1
    for _mn in ("Minimax", "MMWV"):
        _m = METHODS[_mn]
        assert manipulation(_m, _m.sincere(_ue), _ue, _ue, lambda P: P)["manip2"] == 0, _mn
assert _n_cw >= 15
print("manipulation ok")

# Equal-ranking variants equal their strict versions on sincere (strict) ballots
r = np.random.default_rng(11)
pairs = [("Borda", "Borda-ER"), ("Baldwin", "Baldwin-ER"), ("Minimax", "Minimax-ER"),
         ("Schulze", "Schulze-ER"), ("Copeland//Margins-ER", "Copeland//Margins"), ("Minimax", "MMWV")]
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


def _p(**change):
    """Run parameters for one_election in the tests (the keys of run.PARAMS)."""
    return {"seed": 3, "samples": 5, "group_size": 10, "strategic": 0.5, "max_score": 5, "exits": 0, "clones": 0,
            "poll_size": 0, "poll_error": 2.0, "search": 2, "converge": 0, "aggressive": 0.5, "start": "honest",
            "lazy": 0.0, **change}


rows = []
for e in range(6):
    rows += one_election({"sc": Scenario(n_voters=31, n_cands=5), "p": _p(exits=1, clones=2), "e": e, "rounds": [1],
                          "names": ["Borda", "Borda-ER", "IRV", "ER-IRV"]})
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
bs = METHODS["Score runoff"].sincere(r.random((1, 5)))[0]
for _ in range(50):
    nb = _mut(bs, "score_run", r)
    assert np.array_equal(nb % 5, bs % 5) and 0 <= (nb // 5).min() and (nb // 5).max() <= _methods_mod.MAXS
print("two-round elections: strategy changes only round-1 marks ok")

# Score runoff: top two by score, then the separate runoff by the sincere rankings. 2 voters A5 B5 C0
# (A > B), 1 A0 B5 C4, 1 A2 B0 C5: totals A12 B15 C9, B and A go on. The runoff rankings give A 3:1;
# STAR's runoff reads the scores, where the first two voters rate A and B equally: 1:1, B by score.
sr = profile([(2, [1.0, 0.97, 0.0]), (1, [0.0, 1.0, 0.9]), (1, [0.4, 0.0, 1.0])])
Bs = METHODS["Score runoff"].sincere(sr)
assert np.array_equal(Bs // 3, METHODS["Score"].sincere(sr)) and np.array_equal(Bs % 3, METHODS["Borda"].sincere(sr))
check("Score runoff (runoff by rankings: A)", METHODS["Score runoff"].tally(Bs), [1, 0, 0])
check("STAR on the same ballots (equal scores: B)", METHODS["STAR"].honest(sr), [0, 1, 0])

# DMS: 4 x A5 B4 C0, 4 x C5 B4 A0, 1 x A5 C4 B0. Score elects B (32, though nobody's favourite);
# DMS: B has 0 at depth 1; A is above C at every depth (lowest values A 1.67, C 1.6): A.
Bd = np.array([[5, 4, 0]] * 4 + [[0, 4, 5]] * 4 + [[5, 0, 4]], dtype=float)
check("Score on the dark-horse profile", METHODS["Score"].tally(Bd), [0, 1, 0])
check("DMS on the dark-horse profile", METHODS["DMS"].tally(Bd), [1, 0, 0])
# ESR: points from the top k levels, per depth (columns A, B, C): A5 B5 C1, A2 B1 C4, A3 B5 C4,
# A5 B5 C5, A0 B0 C2, A2 B5 C4, A2 B3 C3. B has the most points at every depth (20, 20, 23, 23, 24):
# B alone in the set, B wins. With voters instead of points (the Buck-set) A, B and C would all
# enter and the stretched runoff would elect C.
Be = np.array([[5, 5, 1], [2, 1, 4], [3, 5, 4], [5, 5, 5], [0, 0, 2], [2, 5, 4], [2, 3, 3]], dtype=float)
check("ESR: one candidate most supported at every depth", METHODS["ESR"].tally(Be), [0, 1, 0])
# a set of two: 2 x A5 B4 C0, 1 x B5 A0 C3. Points at depth 1: A 10, B 5; depth 2: A 10, B 13; the
# set is A, B; stretched over it: A5 B0 twice, B5 A0 -> A 10, B 5: A, where Score elects B (13 > 10)
Be2 = np.array([[5, 4, 0], [5, 4, 0], [0, 5, 3]], dtype=float)
check("ESR: stretched runoff of the set", METHODS["ESR"].tally(Be2), [1, 0, 0])
check("Score on the same ballots", METHODS["Score"].tally(Be2), [0, 1, 0])

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
    assert sincerity(mi, mi.sincere(u), u).min() == 1                    # honest = sincere for its own ballot
    assert np.allclose(sincerity(mi, mo.sincere(u), u), 0)              # the original's honest ballot: as far against them as it gets here
check("Inverse-Minimax elects the Condorcet loser", METHODS["Inverse-Minimax"].honest(u), [0, 0, 0, 1])
check("Inverse-Plurality elects the most often least liked", METHODS["Inverse-Plurality"].honest(u), [0, 0, 0, 1])
# Inverse-TTR: marking the favourite in round 1 gives TTR (the runoff reads the reversed ranking back)
Bt = METHODS["Inverse-TTR"].sincere(u)
assert sincerity(METHODS["Inverse-TTR"], Bt, u).min() == 1
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

# Combining runs (analyze.combine) relies on each method's result depending only on the election,
# not on which other methods ran: a method alone gives the same rows as in company.
def _run(names):
    out = []
    for e in range(3):
        out += one_election({"sc": Scenario(n_voters=31, n_cands=4), "p": _p(seed=5, search=3), "e": e,
                             "rounds": [1, 2], "names": names})
    return {(r["election"], r["rounds"], r["ballots"]): (r["U_method"], r["sincerity"]) for r in out if r["method"] == "STAR"}
assert _run(["STAR"]) == _run(["Plurality", "STAR", "Minimax"])
print("method results independent of the other methods ok")

import analyze as _an
def _rows(f, methods, elections, u=1.0):
    base = {k: "1" for k in _an.PARAMS}
    return [{**base, "seed": "1", "election": str(e), "method": m, "ballots": b, "_file": f, "U_best": str(u + e),
             "U_rb": "0", "WM_best": "1", "WM_rb": "0", "U_method": str(len(m)), "WM": "0", "sincerity": "1"}
            for e in elections for m in methods for b in ("sincere", "learned")]
rs, notes, cover = _an.combine(_rows("a", ["X", "Y"], range(4)) + _rows("b", ["Y", "Z"], range(3)))
assert {r["method"] for r in rs} == {"X", "Y", "Z"} and {r["election"] for r in rs} == {"0", "1", "2"}
assert all(r["_file"] == "b" for r in rs if r["method"] == "Y") and len(notes) == 1 and len(cover) == 1
try:
    _an.combine(_rows("a", ["X"], range(2)) + _rows("b", ["Y"], range(2), u=2.0))
    raise AssertionError("different electorates were combined")
except SystemExit:
    pass
print("combine ok")

# Sincerity (reversals within the format) and fidelity: utilities 0.5 / 0.3 / 0.15 / 0.05 (docs/analysis.md, Sincerity and fidelity)
from strategy import sincerity, fidelity, _max_separated
from itertools import product as _prod
u1 = np.array([[0.5, 0.3, 0.15, 0.05]])
for name, B, sin, fid in [("Plurality", [1, 0, 0, 0], 1.0, 0.717), ("Plurality", [0, 1, 0, 0], 0.75, 0.317),
                          ("Plurality", [0, 0, 0, 1], 0.0, 0.05), ("Borda", [2, 3, 1, 0], 0.867, 0.633),
                          ("Borda", [0, 1, 2, 3], 0.0, 0.0), ("Score", [5, 3, 1, 0], 1.0, 0.97),
                          ("Score", [5, 5, 0, 0], 1.0, 0.7), ("Approval", [1, 1, 0, 0], 1.0, 0.7),
                          ("Approval", [0, 0, 1, 1], 0.0, -0.1),
                          ("TTR", [3, 2 + 4, 1, 0], 0.75, 0.317)]:   # TTR: round-1 mark on the 2nd choice = plurality
    X = np.array([B], float)
    got = (float(sincerity(METHODS[name], X, u1)[0]), float(fidelity(METHODS[name], X, u1)[0]))
    assert abs(got[0] - sin) < 1e-3 and abs(got[1] - fid) < 2e-3, (name, B, got, sin, fid)
for name in METHODS:                                               # sincere ballots never reverse a preference
    m = METHODS[name]
    uu = r.random((20, 5))
    assert np.allclose(sincerity(m, m.sincere(uu), uu), 1), name
# the worst ballot with k levels (dynamic programme) = brute force over all ballots with k levels
for k in (2, 3):
    for _ in range(10):
        Pk = r.random((1, 5))
        w = np.clip(Pk[:, :, None] - Pk[:, None, :], 0, None)
        best = max((w * (np.array([X])[:, None, :] > np.array([X])[:, :, None])).sum()
                   for X in _prod(range(k), repeat=5))
        assert abs(best - _max_separated(Pk, k)[0]) < 1e-9
from methods import sincerity_defined
assert not sincerity_defined("TRASH") and not sincerity_defined("Inverse-Plurality") and sincerity_defined("STAR")
print("sincerity and fidelity ok")

# Condorcet winner in the cast ballots: a sincere ranking shows the real one; burying it behind
# a rival in enough ballots makes the ballots show a false one
from metrics import ballot_condorcet
u2 = profile([(4, [0.5, 0.3, 0.15, 0.05]), (3, [0.15, 0.3, 0.5, 0.05]), (2, [0.3, 0.5, 0.15, 0.05])])
Bor = METHODS["Borda"]
d = ballot_condorcet(Bor, Bor.sincere(u2), u2, lambda P: P, Bor.honest(u2))
assert d["bcw"] == "same", d
Bb = Bor.sincere(u2).copy()
Bb[:4] = [3, 1, 2, 0]                                  # the 4 A-voters put C above B (the real CW is B)
assert ballot_condorcet(Bor, Bb, u2, lambda P: P, Bor.tally(Bb))["bcw"] in ("false", "none")
assert ballot_condorcet(METHODS["Plurality"], METHODS["Plurality"].sincere(u2), u2, lambda P: P, np.ones(4) / 4)["bcw"] == ""
print("ballot Condorcet winner ok")

# Truncated rankings (unranked = 0, tied last) and laziness
import methods as _mm
from strategy import mutate as _mutate
Bt = np.array([[0, 0, 2]] * 3 + [[2, 1, 0]] * 4 + [[1, 2, 0]] * 4, dtype=float)   # 3 bullet votes for C
check("IRV: bullet ballots are exhausted once their candidate is out", METHODS["IRV"].tally(Bt), [0.5, 0.5, 0])
check("ER-IRV: the same with level-0 ties", METHODS["ER-IRV"].tally(Bt), [0.5, 0.5, 0])
assert np.allclose(METHODS["Borda"].features(np.array([[3., 0, 0, 0]])), [[3, 0, 0, 0]])   # unranked: 0 points
assert np.allclose(_mm.FEATURES["dowdall"][0](np.array([[3., 0, 0, 0], [3., 2, 1, 0]])),
                   [[12, 0, 0, 0], [12, 6, 4, 3]])                                          # unranked: 0 (lcm 12)
L = _mm.FEATURES["lasts"][0](np.array([[3., 0, 0, 0]])).reshape(16, 4)
assert np.allclose(L[15], [0, 1 / 3, 1 / 3, 1 / 3])                                         # Coombs: last place shared
FS = _mm.FEATURES["firstsecond"][0](np.array([[3., 0, 0, 0], [3., 2, 0, 0]])).reshape(2, 4, 4)
assert FS[0, 0, 0] == 1 and FS[1, 0, 1] == 1                                                 # bullet: no second preference
import ballots as _bl
ul = np.array([[1.0, 0.6, 0.3, 0.0]])
assert np.allclose(_bl.lazy_rank_ballots(ul, 0.0), [[3, 2, 1, 0]])
assert np.allclose(_bl.lazy_rank_ballots(ul, 0.5), [[3, 2, 0, 0]])
assert np.allclose(_bl.lazy_rank_ballots(ul, 1.0), [[3, 0, 0, 0]])
assert np.allclose(_bl.lazy_rank_ballots(ul, 0.3), [[3, 2, 1, 0]])                           # all but one = full ranking
_mm.set_lazy(1.0)
assert np.allclose(METHODS["IRV"].sincere(ul), [[3, 0, 0, 0]])
assert np.allclose(METHODS["MDDA"].sincere(ul), [[1, 0, 0, 0]])                                # trunc: lazy cuts further
assert np.allclose(METHODS["Inverse-IRV"].sincere(ul), [[0, 1, 2, 3]])                         # inverse: never lazy
assert np.allclose(METHODS["Approval"].sincere(ul), [[1, 0, 0, 0]])                            # bullet approval
assert np.allclose(METHODS["Score"].sincere(ul), [[5, 0, 0, 0]])                               # others blank = 0
assert np.allclose(METHODS["MCA"].sincere(ul), [[2, 0, 0, 0]])
assert np.allclose(METHODS["Inverse-Approval"].sincere(ul), [[0, 0, 1, 1]])                   # inverse: never lazy
for name in METHODS:                                                                           # lazy honest ballots never reverse
    m = METHODS[name]
    uu = r.random((20, 5))
    if sincerity_defined(name):
        assert np.allclose(sincerity(m, m.sincere(uu), uu), 1), name
_mm.set_lazy(0.0)
rg = np.random.default_rng(3)
b = np.array([4., 3, 2, 1, 0])
for _ in range(300):                                                                           # mutations stay valid rankings
    b = _mutate(b, "rank", rg)
    pos = np.sort(b[b > 0])[::-1]
    assert np.allclose(pos, np.arange(4, 4 - len(pos), -1)) and 1 <= len(pos) <= 4
# per-voter laziness: each row with its own cut, the same as that row alone with that cut
_ul = np.random.default_rng(14).random((6, 5))
_cuts = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
from ballots import BALLOTS as _BL
for _bt, _b in _BL.items():
    if _b.lazy:
        _all = _b.sincere(_ul, _cuts)
        for _i, _c in enumerate(_cuts):
            assert np.array_equal(_all[_i], _b.sincere(_ul[_i:_i + 1], _c)[0]), (_bt, _i)
_mm.set_voter_lazy(_cuts)
assert np.array_equal(METHODS["IRV"].sincere(_ul), _BL["rank"].sincere(_ul, _cuts))
try:
    METHODS["IRV"].sincere(_ul[:2])
    raise AssertionError("per-voter laziness with the wrong number of rows must fail")
except ValueError:
    pass
assert abs(_mm.lazy_of(np.array([1, 2])) - 0.3) < 1e-12
_mm.set_voter_lazy(None)
print("truncated rankings and laziness ok")

# Ballot types (ballots.py, also those from add-ons): every entry is complete and consistent
from ballots import BALLOTS as _BT
from strategy import worst_reversal as _wr, implied_values as _iv, exaggerations as _ex
_rb = np.random.default_rng(11)
for _name, _b in _BT.items():
    for C in (2, 3, 5):
        _u = _rb.random((30, C))
        _m = Method(_name, d_minimax)                       # any decide: only the ballot type matters here
        _B = _m.sincere(_u)
        assert _B.shape == _u.shape, _name
        _v = _iv(_m, _B)
        assert _v.min() >= -1e-9 and _v.max() <= 1 + 1e-9, _name                 # implied values on 0..1
        assert (_wr(_m, _u) >= -1e-9).all(), _name
        for _row in _B[:5]:
            _x = _b.mutate(_row.copy(), _rb)
            assert _x.shape == _row.shape, _name
        for _x in _ex(_name, _u[:4].sum(axis=0), 0, 1):
            assert _x.shape == (C,), _name
print(f"ballot types ok: {', '.join(_BT)}")

# Method settings (methods.SETTINGS): a method may fix the score range for itself (named Name-X);
# every other method follows the run's range (--max-score)
_u = np.random.default_rng(5).random((40, 4))
for _g in (5, 10):
    _mm.set_max_score(_g)
    for _n, _m in METHODS.items():
        if _m.ballot == "score" and not _m.settings and not _m.inverse:
            assert _m.sincere(_u).max() == _g, _n                         # base methods: the run's range
    for _n, _m in METHODS.items():
        if (_m.settings or {}).get("max_score") is not None:
            assert _n.endswith(f"-{_m.settings['max_score']}"), f"{_n}: a fixed range is named Name-X"
_mm.set_max_score(5)
METHODS["Score-10"] = Method("score", _mm.d_sum, ("sum",), settings={"max_score": 10})
assert METHODS["Score-10"].sincere(_u).max() == 10 and _mm.MAXS == 5           # own range, then back
_jp = _p(seed=4, search=3)
_fixed = one_election({"sc": Scenario(n_voters=21, n_cands=4), "p": {**_jp, "max_score": 5}, "e": 0, "rounds": [1, 2],
                       "names": ["Score-10", "STAR"]})
_glob = one_election({"sc": Scenario(n_voters=21, n_cands=4), "p": {**_jp, "max_score": 10}, "e": 0, "rounds": [1, 2],
                      "names": ["Score"]})
_strip = lambda rows, n: [{k: v for k, v in r.items() if k != "method"} for r in rows if r["method"] == n]
assert _strip(_fixed, "Score-10") == _strip(_glob, "Score")      # Score-10 at --max-score 5 = Score at --max-score 10
_mm.set_max_score(5)                                             # one_election sets the run's range
del METHODS["Score-10"]
print("method settings ok")

# Electorate models (models.py)
import models as _md
_sk = Scenario(n_voters=51, n_cands=5, model="ks")
for _e in range(20):
    _el = _md.draw_electorate(_sk, np.random.default_rng([3, _e]))
    assert _el.u.shape == (51, 5) and (_el.u > 0).all() and (_el.u <= 1).all()
    assert np.allclose(np.linalg.norm(_el.voter_weights, axis=1), 1)            # weights normalised per person
    assert _el.voters.shape[1] == _el.voter_weights.shape[1] == _el.cands.shape[1]
_a = _md.draw_electorate(_sk, np.random.default_rng([3, 0])).u
assert np.array_equal(_a, _md.draw_electorate(_sk, np.random.default_rng([3, 0])).u)   # reproducible
# ks parameters do not touch the clusters model
_c1 = _md.draw_electorate(Scenario(n_voters=21, n_cands=4, model="clusters"), np.random.default_rng(9)).u
_c2 = _md.draw_electorate(Scenario(n_voters=21, n_cands=4, model="clusters", ks_alpha=5.0, ks_cut=0.5),
                          np.random.default_rng(9)).u
assert np.array_equal(_c1, _c2)
# faction alpha: tiny -> (almost) one faction, every person close to it; huge -> everyone on their own
def _spread(alpha):
    el = _md.draw_electorate(Scenario(n_voters=201, n_cands=3, model="ks", ks_alpha=alpha), np.random.default_rng(4))
    return el.voters.std(axis=0).mean()
assert _spread(1e-6) < _spread(1e6)
# ks_shared: the same factions in every bundle; without it, factions differ between bundles
for _sh in (0, 1):
    _f = []
    for _e in range(10):
        _f += [_md._ks_structure(Scenario(n_voters=51, n_cands=4, model="ks", ks_shared=_sh), np.random.default_rng([6, _e]))[2]]
    _same = [all(np.array_equal(fs[0], x) for x in fs[1:]) for fs in _f if len(fs) > 1]
    assert _same and (all(_same) if _sh else not all(_same)), _sh
# ks-quinn: one offset per person and bundle, so two members of a faction differ by the same amount on
# every dimension of that bundle; in ks they do not
for _q, _expect in ((True, True), (False, False)):
    _hits = []
    for _e in range(30):
        _pos, _, _facs, _bund = _md._ks_structure(Scenario(n_voters=51, n_cands=4, model="ks"),
                                                  np.random.default_rng([8, _e]), quinn=_q)
        _start = 0
        for _fac, _k in zip(_facs, _bund):
            if _k >= 2:
                _i, _j = np.flatnonzero(_fac == np.bincount(_fac).argmax())[:2]
                _d = _pos[_i, _start:_start + _k] - _pos[_j, _start:_start + _k]
                _hits.append(bool(np.allclose(_d, _d[0])))
            _start += _k
    assert _hits and all(_h == _expect for _h in _hits), _q
print("electorate models ok")

# Sweep order of parameter values (analyze.param_order): numbers by value, words in their listed order
from analyze import param_order as _po
assert sorted(["ks", "clusters"], key=lambda v: _po("model", v)) == ["clusters", "ks"]
assert sorted(["0", "5", "1"], key=lambda v: _po("group_size", v)) == ["1", "5", "0"]
assert sorted(["previous", "honest"], key=lambda v: _po("start", v)) == ["honest", "previous"]
print("parameter order ok")

# Space from approval ballots (show_electorates.sav_space): candidates on a line give one dominant
# dimension; candidates on a circle give two equal ones
import show_electorates as _se
_r = np.random.default_rng(12)
_x = _r.normal(0, 1, (4000, 1))
_line = np.linspace(-1.5, 1.5, 7)[None, :]
_A = (_r.random((4000, 7)) < np.exp(-0.5 * (_x - _line) ** 2)).astype(float)
_sh, _npos, _zero = _se.sav_space(_A)
assert _sh[0] > 0.6 and _zero == 0, _sh
_ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
_cand = np.c_[np.cos(_ang), np.sin(_ang)]
_v = _r.normal(0, 1, (6000, 2))
_d2 = ((_v[:, None, :] - _cand[None]) ** 2).sum(axis=2)
_sh, _, _ = _se.sav_space((_r.random(_d2.shape) < np.exp(-0.7 * _d2)).astype(float))
assert abs(_sh[0] - _sh[1]) < 0.1 and _sh[:2].sum() > 0.6, _sh
print("space from approval ballots ok")

# clusters model options: salience per cluster leaves positions unchanged and adds weights (mean square 1,
# shared within a cluster); fixed total variance scales positions by one factor
_b = Scenario(n_voters=41, n_cands=4, model="clusters", dims=6, n_clusters=3, zipf=0, salience="none",
              salience_exponent=1.0, fixed_variance=0)
_e0 = _md.draw_electorate(_b, np.random.default_rng(21))
_e1 = _md.draw_electorate(Scenario(**{**_b.__dict__, "salience": "cluster"}), np.random.default_rng(21))
assert np.array_equal(_e0.voters, _e1.voters) and _e0.voter_weights is None
assert np.allclose((_e1.voter_weights ** 2).mean(axis=1), 1)
assert len({tuple(np.round(w, 9)) for w in _e1.voter_weights}) <= 3                    # one weighting per cluster
assert sorted(np.round(_e1.voter_weights[0] / _e1.voter_weights[0].max(), 6)) == sorted(np.round(1 / np.arange(1, 7), 6))
_e2 = _md.draw_electorate(Scenario(**{**_b.__dict__, "fixed_variance": 1}), np.random.default_rng(21))
assert np.allclose(_e2.voters, _e0.voters * np.sqrt(np.pi ** 2 / 6 / 6))
_e3 = _md.draw_electorate(Scenario(**{**_b.__dict__, "salience": "cluster", "salience_exponent": 0.0}),
                          np.random.default_rng(21))
assert np.allclose(_e3.voter_weights, 1) and np.allclose(_e3.u, _e0.u)            # exponent 0: all alike = none
print("clusters options ok")

# field model: reproducible, positive utilities, fixed total variance; large saturation pushes each issue
# towards two camps; the salience field gives per-person weights with mean square 1
_sf = Scenario(n_voters=301, n_cands=5, dims=8, model="field")
_f1 = _md.draw_electorate(_sf, np.random.default_rng(31))
assert np.array_equal(_f1.u, _md.draw_electorate(_sf, np.random.default_rng(31)).u)
assert (_f1.u > 0).all() and _f1.voter_weights is None
assert abs((_f1.voters ** 2).mean() * 8 - np.pi ** 2 / 6) < 0.15
def _bimodal(k):
    v = _md.draw_electorate(Scenario(**{**_sf.__dict__, "field_kappa": k}), np.random.default_rng(31)).voters
    v = v / np.abs(v).max()
    return np.mean(np.abs(v) > 0.8)
assert _bimodal(20.0) > 0.8 > 0.3 > _bimodal(0.3)
_f2 = _md.draw_electorate(Scenario(**{**_sf.__dict__, "field_salience": 1.0}), np.random.default_rng(31))
assert np.allclose((_f2.voter_weights ** 2).mean(axis=1), 1)
print("field model ok")

# noise model: reproducible, positive utilities, distinct people, total variance pi^2/6; strength 0 is a
# Gaussian blob (variance spread evenly over the dimensions), a high strength concentrates it on fewer directions
_sn = Scenario(n_voters=301, n_cands=5, dims=6, model="noise")
_n1 = _md.draw_electorate(_sn, np.random.default_rng(41))
assert np.array_equal(_n1.u, _md.draw_electorate(_sn, np.random.default_rng(41)).u)
assert (_n1.u > 0).all() and _n1.voter_weights is None
_people = np.vstack([_n1.voters, _n1.cands])
assert len(np.unique(_people.round(9), axis=0)) >= len(_people) - 3
assert abs((_people ** 2).sum(axis=1).mean() - np.pi ** 2 / 6) < 0.4
def _pc2(strength):
    out = []
    for e in range(10):
        v = _md.draw_electorate(Scenario(**{**_sn.__dict__, "noise_strength": strength}), np.random.default_rng([42, e])).voters
        ev = np.sort(np.linalg.eigvalsh(np.cov(v.T)))[::-1]
        out.append(ev[:2].sum() / ev.sum())
    return np.median(out)
assert _pc2(0.0) < 0.5 < _pc2(8.0), (_pc2(0.0), _pc2(8.0))
print("noise model ok")

# Renamed methods and groups: old names lead to existing current ones, in selection and in analysis
for _old in _mm.RENAMED:
    assert _mm.current_name(_old) in METHODS, _old
    assert _mm.select([_old]) == [_mm.current_name(_old)], _old
for _old in _mm.RENAMED_GROUPS:
    assert _mm.select([_old]) == _mm.select([_mm.current_name(_old, _mm.RENAMED_GROUPS)]), _old
assert _mm.current_name("IRV") == "IRV"
import analyze as _an, csv as _csv, tempfile as _tf, os as _os
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "regression.csv"), newline="") as _f:
    _old_rows = [r for r in _csv.DictReader(_f) if r["method"] == "TTR"]
for r in _old_rows:
    r["method"] = "Top-two runoff"                                     # as a run from before 2026-10-07
with _tf.TemporaryDirectory() as _d:
    _p = _os.path.join(_d, "elections.csv")
    with open(_p, "w", newline="") as _f:
        _w = _csv.DictWriter(_f, list(_old_rows[0]))
        _w.writeheader()
        _w.writerows(_old_rows)
    assert {r["method"] for r in _an.load([_p])} == {"TTR"}
    assert _an.RENAMED_SEEN == {"Top-two runoff": "TTR"}
# the standard set: existing methods, the default selection, overlapping the groups
from methods import COMPARISONS as _CMP, run_selection as _rs
assert all(m in METHODS for m in _CMP["standard"]) and _rs() == [m for m in METHODS if m in _CMP["standard"]]
assert _rs(exclude=["Plurality"]) == [m for m in _rs() if m != "Plurality"] and "Kemeny-Young" in _rs(["Kemeny-Young"])
print("renamed methods ok")

# Honest approval: a personal threshold between the voter's mean and favourite (ballots.approves)
from ballots import approves as _apv
_ua = np.array([[1.0, 0.8, 0.5, 0.2, 0.0]])                          # stretched already; mean 0.5
assert _apv(_ua, 0.0).tolist() == [[True, True, True, False, False]]   # draw 0: down to the mean
assert _apv(_ua, 1.0).tolist() == [[True, False, False, False, False]] # draw 1: the favourite only
assert _apv(_ua, 0.5).tolist() == [[True, True, False, False, False]]  # halfway: threshold 0.75
_d = np.linspace(0, 1, 1001)                                        # chance rises linearly from mean to favourite
assert abs(_apv(np.repeat(_ua, len(_d), 0), _d)[:, 1].mean() - 0.6) < 0.01   # (0.8 - 0.5) / (1 - 0.5)
_methods_mod.set_voter_approval([0.0, 1.0])                         # per voter, through Method.sincere
_two = np.array([[0.9, 0.6, 0.3, 0.0], [0.9, 0.6, 0.3, 0.0]])
assert METHODS["Approval"].sincere(_two).tolist() == [[1, 1, 0, 0], [1, 0, 0, 0]]
assert METHODS["MDDA"].sincere(_two).tolist() == [[2, 1, 0, 0], [1, 0, 0, 0]]    # trunc: ranks the approval set
assert abs(_methods_mod.approval_of(np.array([0, 1])) - 0.5) < 1e-12
try:
    METHODS["Approval"].sincere(_two[:1]); raise AssertionError("row mismatch not caught")
except ValueError:
    pass
_methods_mod.set_voter_approval(None)
print("approval threshold ok")

# Add-ons (addons/*.py): each file's own test(), and the template as a check of the loader contract
import methods as _ma, importlib.util as _iu, os as _os
for _fn in sorted(set(_ma.ADDONS.values())):
    _mod = __import__("sys").modules["bffl_addon_" + _fn[:-3]]
    if hasattr(_mod, "test"):
        _mod.test()
    print(f"add-on {_fn} ok: {', '.join(k for k, v in _ma.ADDONS.items() if v == _fn)}")
_spec = _iu.spec_from_file_location("bffl_template", _os.path.join(_ma.ADDON_DIR, "_template.py"))
_tm = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
_ma.FEATURES.update(_tm.FEATURES)
_tm.test()
del _ma.FEATURES["my_feature"]
print("add-on template ok")

# Utility of distance (--utility): linear gives the same positions and the same order of every voter's
# candidates as exp, only other utility differences; fav_ratio is left empty for it.
from scenario import Scenario as _Sc
from models import draw_electorate as _de
from metrics import electorate_stats as _es
for _e in range(5):
    _ex = _de(_Sc(), np.random.default_rng([3, _e]))
    _li = _de(_Sc(utility="linear"), np.random.default_rng([3, _e]))
    assert np.array_equal(np.argsort(-_ex.u, axis=1), np.argsort(-_li.u, axis=1))
    assert np.allclose(_li.u, np.log(_ex.u))                 # -d / width = log exp(-d / width)
    assert _es(_li.u)["fav_ratio"] == "" and _es(_ex.u)["fav_ratio"] != ""
    _qu = _de(_Sc(utility="quadratic"), np.random.default_rng([3, _e]))
    assert np.allclose(_qu.u, -_li.u ** 2)                           # -(d / width)^2
print("linear and quadratic utility ok")

# compare.py: across runs in a folder. Two tiny runs (defaults, linear utility) and a copy of the
# first with another version, which must stop the comparison unless --force-versions is given.
import os, tempfile, subprocess, csv as _csv, run as _run, sys as _sys
_tmp = tempfile.mkdtemp(prefix="bffl-compare-")
_base = {k: v for k, v in _run.default_params().items()}
_base.update(voters=25, rounds=1, samples=3)
_fp = _run.fingerprints(["Score", "Minimax", "STAR"], 0)          # 0: no worker processes (see run.simulate)
assert _fp["Score"] == _run.fingerprints(["Score"], 0)["Score"]          # independent of the other methods
assert len(set(_fp.values())) == 3
for _name, _ch in (("a", {}), ("b", {"utility": "linear"})):
    _rows = _run.simulate({**_base, **_ch}, 6, 0, ["Score", "Minimax", "STAR"], fingerprint=_fp)
    os.makedirs(os.path.join(_tmp, _name), exist_ok=True)
    with open(os.path.join(_tmp, _name, "elections.csv"), "w", newline="") as _f:
        _w = _csv.DictWriter(_f, fieldnames=list(_rows[0]))
        _w.writeheader()
        _w.writerows(_rows)
_cmp = lambda *args: subprocess.run([_sys.executable, "compare.py", _tmp, "--boot", "50", "--min-elections", "1", *args],
                                    capture_output=True, text=True)
_res = _cmp()
assert _res.returncode == 0, _res.stderr
_rep = open(os.path.join(_tmp, "comparison", "report.md")).read()
assert "2 distinct condition(s)" in _rep and "utility linear" in _rep and "| Minimax |" in _rep
with open(os.path.join(_tmp, "comparison", "methods.csv")) as _f:
    assert {r["method"]: r["conditions"] for r in _csv.DictReader(_f)} == {"Score": "2.000", "Minimax": "2.000", "STAR": "2.000"}
assert os.path.exists(os.path.join(_tmp, "comparison", "beatpath.csv")) and "## Clear dominance (beatpath)" in _rep
# bootstrap draws depend only on the elections (and --seed): the same condition gets the same draws in
# analyze.py and compare.py, whatever else is analysed, and the order of the rows does not matter
import analyze as _an
_ids = [(1, 0), (1, 1), (1, 2), (1, 5)]
assert np.array_equal(_an.bootstrap_weights(_ids, 20, 0), _an.bootstrap_weights(list(_ids), 20, 0))
assert not np.array_equal(_an.bootstrap_weights(_ids, 20, 0), _an.bootstrap_weights(_ids[:3] + [(1, 6)], 20, 0))
assert not np.array_equal(_an.bootstrap_weights(_ids, 20, 0), _an.bootstrap_weights(_ids, 20, 1))
with open(os.path.join(_tmp, "a", "elections.csv")) as _f:
    _rows_a = list(_csv.DictReader(_f))
for _r in _rows_a:
    _r["_file"] = "a"
_d1 = _an.analyse_condition(_rows_a, 50, 0)["dominated_by"]
_d2 = _an.analyse_condition(_rows_a[::-1], 50, 0)["dominated_by"]
assert sorted(map(str, _d1)) == sorted(map(str, _d2))
# beatpath: arrows where A dominates B in at least the share, more often than the reverse; arrows implied
# by a chain are kept in the list but left out of the plot
import compare as _cp
_P = {}
for (_x, _y), _d in {("A", "B"): 8, ("B", "C"): 6, ("A", "C"): 9, ("C", "B"): 1, ("D", "A"): 5, ("A", "D"): 5}.items():
    _P[(_x, _y)] = {"shared": 10, "dom": _d}
for _x in "ABCD":
    for _y in "ABCD":
        _P.setdefault((_x, _y), {"shared": 10, "dom": 0})
_all, _kept = _cp.beat_edges(list("ABCD"), _P, 0.5)
assert {(e[0], e[1]) for e in _all} == {("A", "B"), ("B", "C"), ("A", "C")}       # D-A: a tie, no arrow
assert {(e[0], e[1]) for e in _kept} == {("A", "B"), ("B", "C")}
# another version in a third folder: stops, unless forced (then stated in the report)
os.makedirs(os.path.join(_tmp, "c"))
with open(os.path.join(_tmp, "a", "elections.csv")) as _f:
    _rows = list(_csv.DictReader(_f))
for _r in _rows:
    _r["version"] = "0.9.0"
with open(os.path.join(_tmp, "c", "elections.csv"), "w", newline="") as _f:
    _w = _csv.DictWriter(_f, fieldnames=list(_rows[0]))
    _w.writeheader()
    _w.writerows(_rows)
_res = _cmp()
assert _res.returncode != 0 and "different simulator versions" in _res.stderr
_res = _cmp("--force-versions")
assert _res.returncode == 0, _res.stderr
assert "Mixed simulator versions, compared anyway" in open(os.path.join(_tmp, "comparison", "report.md")).read()
# the same version but another fingerprint for Score (the code changed): refused unless forced
for _r in _rows:
    _r["version"] = _run.VERSION
    if _r["method"] == "Score":
        _r["fingerprint"] = "0000000000"
with open(os.path.join(_tmp, "c", "elections.csv"), "w", newline="") as _f:
    _w = _csv.DictWriter(_f, fieldnames=list(_rows[0]))
    _w.writeheader()
    _w.writerows(_rows)
_res = _cmp()
assert _res.returncode != 0 and "fingerprints differ" in _res.stderr and "Score" in _res.stderr
_res = _cmp("--force-versions")
assert _res.returncode == 0 and "Runs of different code, compared anyway" in open(
    os.path.join(_tmp, "comparison", "report.md")).read()
# precautions: a subfolder without run data stops the search; an --out holding other files is refused
os.makedirs(os.path.join(_tmp, "notes"))
_res = _cmp("--force-versions")
assert _res.returncode != 0 and "folders without run data" in _res.stderr and "notes" in _res.stderr
os.rmdir(os.path.join(_tmp, "notes"))
_other = tempfile.mkdtemp(prefix="bffl-other-")
open(os.path.join(_other, "keep.txt"), "w").close()
_res = _cmp("--force-versions", "--out", _other)
assert _res.returncode != 0 and "holds other files" in _res.stderr
assert os.listdir(_other) == ["keep.txt"]
# output folders of run.py / analyze.py: an existing folder with other files is refused (run.check_folder)
_ok = tempfile.mkdtemp(prefix="bffl-out-")
for _n in ("elections.csv", "command.txt", ".DS_Store"):
    open(os.path.join(_ok, _n), "w").close()
os.makedirs(os.path.join(_ok, "analysis"))
_run.check_folder(_ok, _run.RUN_FILES, _run.OWN_DIRS)          # only lab files (and hidden ones): accepted
try:
    _run.check_folder(_other, _run.RUN_FILES, _run.OWN_DIRS)    # keep.txt: refused
    raise AssertionError("check_folder accepted a folder with other files")
except SystemExit as _e:
    assert "keep.txt" in str(_e)
import shutil as _sh
for _d in (_tmp, _other, _ok):                                   # only the folders this test made
    _sh.rmtree(_d)
print("compare.py and output folders ok")
