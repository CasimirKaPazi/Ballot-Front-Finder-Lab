# Voting Pareto Lab — findings

What the simulations have shown so far. The model, parameters and definitions are in [README.md](README.md). Newest first; each section names the settings it used, since the model and defaults changed along the way (plain dimensions before Zipf, CE as an axis before WM). Numbers are point estimates; differences only count where the paired comparison or the noise reference says so.

**Units.** Up to 2026-10-06 the axes were PE (zero = utility-proportional lottery) and WM (the winner's worst margin in % of voters); the sections below keep those units. Since then they are VSE and WME, both with 0 = random ballot and 100 = best candidate (README, Axes). Within one condition the change is a fixed rescaling of each axis (the sums over elections that set the zero and the 100 are the same for all methods), so method order, dominance and rank stability are unaffected; only the numbers differ.

**Names.** On 2026-10-07 some methods were renamed: Top-two runoff → TTR, Approval runoff if needed → Approval cond. runoff (called ACR below), Minimax-wv → MMWV. The sections below use the new names.

## Score range 0–5, 0–10, 0–100 (2026-10-07)

Settings: the defaults (99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5, 400 elections per condition, seed 1; `--max-score 5 10 100`. Methods: Score, MJ, STAR, BTR-score, MARS, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score, and as fixed references 321 and MCA (three grades, unchanged across conditions, as they should be). The electorates are the same in all three conditions.

Honest VSE / WME → learned VSE* / WME*:

| Method | 0–5 | 0–10 | 0–100 |
|---|---|---|---|
| Score | 96.1 / 93.2 → 91.9 / 91.0 | 96.2 / 93.3 → 92.8 / 90.8 | 96.1 / 93.2 → 93.8 / 91.8 |
| Score+ | 95.6 / 92.6 → 94.4 / 93.3 | 96.1 / 94.1 → 95.0 / 94.3 | 96.2 / 94.2 → 94.3 / 94.1 |
| STAR | 95.4 / 95.6 → 94.8 / 96.0 | 94.9 / 97.4 → 93.8 / 97.0 | 94.6 / 99.0 → 94.3 / 97.2 |
| MARS | 95.8 / 95.6 → 94.3 / 96.4 | 95.8 / 97.1 → 94.2 / 97.1 | 95.0 / 99.3 → 93.9 / 98.1 |
| ARA | 96.0 / 94.4 → 94.2 / 95.7 | 95.3 / 96.6 → 94.7 / 95.9 | 94.3 / 97.7 → 93.8 / 96.6 |
| BTR-score | 94.9 / 95.7 → 93.5 / 97.2 | 94.4 / 98.0 → 93.9 / 98.1 | 92.8 / 99.9 → 92.4 / 99.1 |
| Smith//Score | 95.3 / 95.6 → 94.3 / 97.8 | 94.4 / 98.0 → 94.7 / 98.0 | 93.2 / 99.9 → 92.8 / 98.6 |
| MJ | 93.7 / 81.8 → 92.5 / 88.6 | 92.9 / 80.8 → 92.0 / 87.1 | 92.8 / 80.2 → 93.0 / 84.0 |
| Reciprocal score | 92.0 / 85.5 → 91.6 / 91.0 | 91.3 / 83.3 → 93.6 / 87.7 | 90.5 / 80.2 → 92.9 / 86.8 |
| ARA-Top | 91.4 / 88.1 → 91.1 / 88.4 | 91.1 / 90.0 → 90.3 / 90.6 | 89.1 / 90.5 → 88.5 / 88.0 |

Frontier (the same in all conditions): best WME 100 at VSE 92.6, best VSE 100 at WME 88.4, balanced point 97.9 / 98.0.

- **The WME cap of the rated methods was an artefact of the 0–5 range.** With 0–100, BTR-score and Smith//Score reach 99.9 WME with honest ballots and behave like Condorcet methods on full rankings (VSE 92.8–93.2, the best-WME end of the frontier); STAR and MARS reach 99.0–99.3. The "Rated ballots cap WME" observation below holds for 0–5 only.
- **Score:** honest values do not change at all (rounding to 0–5 already averages out over voters). The learned VSE* rises (91.9 → 92.8 → 93.8), WME* stays about level. Not yet checked with a paired comparison across conditions; and since the score mutation differs above 0–5 (README, Strategy engine), part of the change may come from the learning rather than the range.
- **Hybrids move along the frontier rather than toward it:** a wider range trades VSE for WME (STAR 95.4 / 95.6 → 94.6 / 99.0; MARS 95.8 / 95.6 → 95.0 / 99.3). With honest ballots that brings STAR and MARS close to the frontier at the WME end: the straight line from the best-WME end to the balanced point, which the frontier cannot lie below, gives VSE 94.5 at WME 99.3, and MARS has 95.0. Under strategy they stay well inside it (MARS* 93.9 / 98.1).
- **Score+ gains a little** (WME 92.6 → 94.2), consistent with its re-stretching losing less information on a finer range.
- **Methods that suffer from a finer range:** Reciprocal score (honest WME 85.5 → 80.2), MJ (WME* 88.6 → 84.0) and ARA-Top (VSE 91.4 → 89.1). For ARA-Top this is by construction: on 0–100 practically only the favourite gets the top score, so it approaches a plurality-based approval count. Reciprocal score works with factions of top-rated candidates, which shrink to single favourites as well.

## Number of candidates, 3 to 9 (2026-10-07)

Settings: the defaults (99 voters, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5, 400 elections per condition, seed 1; base 3 candidates, swept over 3–9 (`--cands`).

**Electorates.** More candidates make the task harder for every method: a Condorcet winner exists in 99.8% of elections with 3 candidates and 94.8% with 9; it also has the highest total utility in 88.5% and 68.6% of those; preferences get steeper (favourite ÷ least liked, median voter, 1.7 → 3.0).

**Frontier.** The balanced point stays at about 97 VSE / 98 WME throughout. The best-WME end moves between VSE 91 and 94.5 without a trend; the best-VSE end loses some WME (89.7 → 87.5).

Honest VSE / WME → learned VSE* / WME*:

| Method | 3 candidates | 6 candidates | 9 candidates |
|---|---|---|---|
| Plurality | 75 / 83 → 88 / 91 | 52 / 55 → 82 / 75 | 51 / 46 → 78 / 66 |
| TTR | 93 / 92 → 94 / 98 | 87 / 75 → 90 / 90 | 86 / 70 → 90 / 84 |
| IRV | 93 / 92 → 94 / 99 | 90 / 78 → 92 / 92 | 91 / 72 → 91 / 85 |
| ER-IRV | 93 / 92 → 93 / 98 | 90 / 78 → 91 / 90 | 91 / 72 → 93 / 84 |
| STAR | 95 / 98 → 94 / 99 | 95 / 96 → 95 / 96 | 97 / 91 → 96 / 92 |
| MARS | 95 / 99 → 94 / 98 | | 96 / 92 → 96 / 93 |
| Smith//Score | 95 / 98 → 95 / 99 | | 96 / 92 → 95 / 95 |
| Minimax, Schulze, MMPO | VSE 91–94, WME 100, WME* 99–100 at every count | | |

(Empty cells: not taken from the report.)

- **Plurality-based methods lose most** as candidates are added (vote splitting): honest WME drops from 92 to 70–72 for TTR and IRV, and to 46 for Plurality. Learned ballots recover part of it (voters compromise), but not all.
- **Condorcet methods are very stable:** WME 100 with honest ballots at every count and 99–100 learned; VSE stays in the 91–94 band of the best-WME end of the frontier.
- **Hybrids trade WME for VSE:** STAR, MARS and Smith//Score gain about 2 VSE and lose 6–7 WME from 3 to 9 candidates, consistent with the rated-ballot cap (more candidates share a rating on each ballot, hiding more preferences).
- **ER-IRV against IRV:** ER-IRV is more sincere at every count (e.g. 98.1 against 96.4 with 6 candidates, 98.5 against 96.6 with 9). The strategic axes are mixed: VSE* higher with 9 candidates (93.0 against 91.3), lower with 6 (91.4 against 92.1), and WME* often lower at high counts (85.9 against 89.8 with 8). Equal rankings let voters support a compromise without dropping their favourite, which shows in sincerity; it is not a clear gain in outcomes.
- **Rank stability** against 3 candidates: sincerity stays at the noise reference (0.80–0.92 against 0.81). WME and WME* stay high (0.67–0.94) but mostly below their references (0.95, 0.93), so the order shifts a little, in line with the plurality-based methods falling back. With 3 candidates the methods are so close on VSE and VSE* that the noise reference is low (0.29 and −0.01), so the order there is mostly noise.

## MMPO family under full strategy (2026-10-07)

Settings: 200 voters, 6 candidates, otherwise the defaults; strategic share 1.0, 800 elections, seed 1. Methods MMPO, MMWV, Minimax product, MMPOWV.

- All four are identical with honest ballots (VSE 92.3, WME 100, CE 100).
- Learned: VSE* 91.9–92.2, WME* 99.1 (MMPOWV) to 99.5 (Minimax product), CE* 98.2–98.4, sincerity 99.8 for all; burial 1.7–1.9% of voters, compromise 0.5–0.6%, push-over 0.
- **Every paired difference is a tie** (MMPOWV's WME* is 0.2–0.4 below the others, within noise). The slight tendency for MMPOWV seen in an earlier run did not hold up. MMPOWV replaces Minimax product because it is the simpler form (two factors instead of three) with no measurable loss, not because it is better.

## Reachable frontier and combination methods (2026-10-07)

Settings for both runs: the defaults (99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5 and 1.0, seeds 1 and 2.

```
python sweep.py --strategic 0.5 1 --elections 400 --seeds 1 2 --out combo \
    --methods "Minimax+Approval" "Approval product" "Minimax product" ARA STAR BTR-score Smith//Score Minimax Schulze Score MMPO Minimax-wv
python analyze.py combo-elections.csv --out combo-analysis --compare "Approval product:STAR" \
    "Minimax+Approval:STAR" "Approval product:Schulze" "Minimax product:Minimax"
```

(The command uses the method names of the time; three of these methods no longer exist and Minimax-wv is now MMWV.)

**The frontier.** VSE and WME cannot both reach 100: the candidate with the highest total utility is often not the one with the best worst margin. Knowing every voter's sincere utilities, a rule could at best reach (800 elections per condition):

| Point on the frontier | VSE | WME |
|---|---|---|
| best WME | 92.9 | 100 |
| best VSE | 100 | 89.5 |
| most balanced | 97.6 | 98.2 |

The margin-based Condorcet methods sit at the WME end (about 92.9 / 100 with honest ballots). The hybrids (STAR, BTR-score, MARS, ARA, Smith//Score) sit 2–3 points inside the curve, around 95–96 on both. No method comes near the balanced point; the gap is what the methods lose by seeing ballots instead of utilities, and to strategy.

**Combination methods** (proposed in this lab):

- **Minimax product** (margins × winning votes × pairwise opposition, each as a share 0–1): equal to Minimax with honest ballots, a tie at strategic share 0.5, and at full strategy clearly better than Minimax on WME* (+2.1), CE* (+4.5) and sincerity (+1.7). MMPO, MMWV and Minimax product are within each other's noise (WME* at full strategy 99.0, 98.7, 99.1). The variant without the margin factor, **MMPOWV** (pairwise opposition × winning votes), replaces Minimax product; see the next section.
- **Minimax+Approval** (approvals + worst margin) and **Approval product** (winning votes × opposition × approvals, on ARA ballots): undominated, but inside the hybrid cluster. Minimax+Approval ties STAR on everything except WME* at full strategy (+1.8); Approval product trades 0.7 VSE for 0.8 WME against STAR. Neither moves toward the balanced point of the frontier; both were removed.
- **Under full strategy** the Condorcet methods that measure opposition (MMPO, MMWV, Minimax product) are the most honest of all (sincerity 99.6–99.7 against 98.0–98.5 for the hybrids, burial about 3% of voters against 15–22%).

## Margin band, a negative result (2026-10-07)

Margin band (proposed in this lab, removed again): every candidate whose worst margin is within 5% of the voters of the best worst margin is in a band; the band member with the highest score total wins. Same settings as above, 1200 elections per condition, compared with Score, STAR, BTR-score, Smith//Score, MARS, Minimax, Schulze, Ranked pairs and IRV.

- It tied Smith//Score on every axis at both strategy levels: with a Condorcet winner in 98% of elections, the band and the Smith set nearly always pick the same candidate.
- The hoped-for burial resistance did not appear: 17.5% of voters buried at full strategy, about as many as with the other hybrids, against about 6% with Schulze, Minimax and Ranked pairs.
- **Under full strategy the margin-based Condorcet methods weakly dominate the hybrids:** the hybrids' VSE advantage holds only with honest or partly strategic voters. At full strategy all hybrids drop to VSE* 92.2–92.8, level with Schulze, Ranked pairs and Minimax (92.5–93.1), while the Condorcet methods keep the clearly higher WME* (97–98 against 93–95).
- **Rated ballots cap WME** (on 0–5; with 0–100 the cap disappears, see "Score range"): every method taking pairwise preferences from 0–5 ratings (Smith//Score, BTR-score, STAR, Margin band) reached only about 95 WME with honest ballots. Equal ratings hide preferences, so even a Condorcet rule on ratings misses some sincere Condorcet winners.

## Current picture (6 candidates, Zipf, 2026-10-06)

```
python sweep.py --one-at-a-time --zipf 1 --dims 10 --clusters 3 5 --spread 0.5 0.75 \
    --width 1.5 --cands 6 --group-size 10 1 0 --seeds 1 2 --elections 400 --out model \
    --strategic 0.5 1 --rounds 20 --exclude "Condorcet-random-drop" "Reciprocal score"
python analyze.py model-elections.csv --out model-analysis
```

Base: 99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, strategic share 0.5, 20 rounds; 800 elections per condition. These are now the defaults.

Methods that no other method clearly beats on PE, WM, PE* and WM* (out of 6 conditions):

| Method | Undominated in |
|---|---|
| Schulze, MARS | 6 |
| Minimax, Ranked pairs, BTR-score | 5 |
| STAR | 4 |
| all others | never |

- **Two groups lead:** margin-based Condorcet methods (Schulze, Ranked pairs, Minimax) and score runoffs (MARS, BTR-score, STAR). Score runoffs are ahead on PE, the Condorcet methods on WM.
- **Burial sensitivity separates the Condorcet methods.** Their honest values are identical, but Benham, BTR-IRV, SICT and Copeland lose 1–2 points more WM under full strategy than Schulze, Ranked pairs and Minimax.
- **Plurality-based methods gain from strategy, but never enough.** Under full strategy TTR's WM rises from 6.5 (honest) to 13.7 and IRV's from 7.3 to 13.2, while Score drops from 13.0 to 10.9; the margin-based Condorcet methods lose least (15.7 → 14.5–14.9).
- **What changes the method order:** full strategy (PE* rank agreement 0.46 against 0.79 from noise), 5 clusters (0.56) and group size 1 (0.58). Spread 0.75 does not.
- **Cycles stay rare** with 6 candidates too (1.6–3% of elections), so how a method resolves cycles matters less here than how well it resists burial.
- **Group size changes little with 6 candidates,** partly for a mechanical reason: groups are voters with the same full ranking, and with 720 possible rankings most groups are small whatever the cap (see README, Coordination).

The same kind of sweep with 4 candidates (width 1.0 1.5 2.0, spread 0.5 0.75 1.0, clusters 3 2 5, strategic share 0.5, 800 elections per condition) gave a similar but wider set: Schulze (undominated in 6 of 7 conditions), STAR, MARS, BTR-Borda (5), Minimax (4), BTR-score, SICT, Baldwin, Approval runoff (2), ARA, Score (1). Width hardly changed the order there; spread and 2 clusters did, mainly under strategy.

## Voter model (2026-10-05/06)

**Plain dimensions, one parameter at a time** (99 voters, 4 candidates, group size 10, strategic share 0.5, 15 rounds, 200 elections per condition): most apparent reordering was within sampling noise, especially on PE*. The electorates themselves (1500–2000 generated per setting) explained most of the rest:

| Setting | Condorcet winner also highest total utility | PE lost electing it | Favourite ÷ least liked (median) |
|---|---|---|---|
| base (2 dims, width 1) | 80% | 7.7% | 3.9 |
| dims 4 | 76% | 10.9% | 6.0 |
| width 0.5 | 74% | — | 17 |
| width 2.0 | 85% | — | 2.0 |

(The PE-lost figure is a ratio of sums, as in the report.) More dimensions and narrower width make the Condorcet winner less often the best candidate by total utility, which lowers PE for Condorcet and plurality-based methods and favours rated methods. Width 0.5 makes preferences implausibly steep (a voter values the favourite 17 times the least liked candidate).

**Zipf dimensions** (dimension i with standard deviation 1/i, after Clay Smith) make the number of dimensions almost irrelevant: from about 4 dimensions on the electorates are the same within noise.

| Model (width 1) | Condorcet winner also highest total utility | PE lost electing it | Favourite ÷ least liked |
|---|---|---|---|
| plain, 2 dims | 80% | 7.7% | 3.9 |
| Zipf, 4 dims | 80% | 7.0% | 2.9 |
| Zipf, 10 dims | 77% | 8.7% | 2.8 |
| Zipf, 20 dims | 79% | 8.5% | 2.8 |

In a direct comparison (200 elections each), Zipf kept the method order on WM and on PE at width 1, widened the PE gap between rated and Condorcet methods by 1–2 points, and hit Plurality hardest (PE 81 → 71). With Zipf the methods cluster more clearly by type.

## Approval with conditional runoff (2026-10-05)

In Germany a winner is expected to have more than half the votes. The tempting chain "use approval instead of TTR, and add a runoff only when nobody has a majority" does not work in the model.

Settings: 600 elections, 200 voters, 6 candidates, plain 2 dimensions, 3 clusters, spread 0.5, width 1.0, group size 10, full strategy, 20 rounds (CE was still an axis then). Paired differences, percentage points (ties in brackets are within noise):

| Pair | PE | CE | PE* | CE* | dissonance |
|---|---|---|---|---|---|
| ACR − Approval | tie | tie | tie | tie | +2.8 |
| Approval runoff − Approval | +2.0 | +19.2 | tie | +9.4 | +1.4 |
| Approval runoff − TTR | +3.3 | +20.4 | +1.5 | tie | −1.3 |
| ACR − Approval runoff | −2.2 | −19.2 | −1.6 | −9.0 | +1.4 |
| ACR − TTR | tie | tie | tie | −7.5 | tie |

("Dissonance" was the complement of today's sincerity.)

- With honest ballots the most approved candidate had more than half the approvals in 97% of elections, and the top two both did in 84%. So the conditional runoff almost never happens: ACR is Approval, plus a threshold that invites strategy.
- A majority of first choices implies a majority against every rival; a majority of approvals does not. The 50% rule checks for a real majority in plurality, but almost nothing in approval.
- Approval with an obligatory runoff is better than TTR on three of four axes, ties on CE*, and has less dissonance. If a 50%-winner is required, that is the version to use.
- Conditional TTR needs no separate test: in the model it is identical to TTR (README, Strategy engine).

## Coordination: burial against Condorcet methods (2026-10-03)

Plain 2 dimensions, 99 voters, 4 candidates, full strategy, 60 elections; CE with learned ballots:

| Method | Whole ranking groups | Every voter alone | Burial (groups / alone) |
|---|---|---|---|
| TTR | 94.9 | 94.1 | 0.1% / 0.3% |
| IRV | 96.6 | 91.5 | 1.8% / 1.0% |
| Copeland | 83.6 | 100.0 | 20.1% / 2.0% |
| Minimax | 89.8 | 100.0 | 20.6% / 1.3% |
| BTR-Borda | 91.5 | 100.0 | 23.2% / 1.2% |
| BTR-score | 89.8 | 96.6 | 20.5% / 2.8% |
| STAR | 89.8 | 98.3 | 18.6% / 3.1% |

Burial needs blocs; compromise works for individuals. With organised factions, Condorcet methods and score runoffs lose much of their Condorcet efficiency to burial, while TTR and IRV are carried toward the Condorcet winner by compromise. With individual voters the picture reverses. This is why every strategic result must be read with its group size.

## Strategy engine checks (2026-10-03/04)

- **Rounds:** with 50 voters and 6 candidates, PE*, CE* and sincerity changed only within noise between 20 and 40 rounds; at 10 rounds they were still moving. 20 rounds is the default.
- **No early stop:** stopping once a round passes without any accepted change was tested (200 elections, 99 voters, 4 candidates, plain 2 dimensions, whole ranking groups, full strategy). It saves most of the run time, but quiet rounds are common long before learning has finished. Stopping after one quiet round changed 55 of 99 method × axis values clearly (sincerity higher for almost every method, CE* up to 9 points higher for Condorcet methods); after two quiet rounds still 50.
- **Honesty wins ties:** without it, deviations that stopped paying off stayed, and dissonance grew with the number of rounds.
