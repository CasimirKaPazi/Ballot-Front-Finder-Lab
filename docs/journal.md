# Ballot Front Finder Lab — development journal

How the lab developed: the experiments, what they showed and the decisions they led to, each with the settings and commands that produced it. Newest first. The software itself (model, parameters, definitions) is described in [README.md](../README.md) and the other files in this folder. Numbers are point estimates; differences only count where the paired comparison or the noise reference says so.

**Changes along the way.** The model, its defaults and the measures changed over time, so each section names its settings. A section uses what was current at its date; the changes, newest first, with what earlier sections used:

| Date | Change | Earlier sections |
|---|---|---|
| 2026-10-26 | Ties recorded per election and method (`metrics.ties`): `split` (the winner drawn by lot) and `zero_margins` (share of candidate pairs the cast ballots leave tied); a Ties table in the report and columns in summary.csv; results unchanged, fingerprints unchanged | not recorded |
| 2026-10-26 | Method set `max-score`: every method that follows `--max-score`, built from the ballot types (`score_range` in `ballots.Ballot`), add-ons included; the reference max-score run uses it (`--methods max-score --exclude inverse`); results unchanged | the max-score run named the groups rated, hybrid and median-based by hand (which included MCA and 321, unaffected by the range) |
| 2026-10-26 | Version 1.1.0. BTR-score and BTR-Borda: a pairwise tie goes to the candidate higher in the order of totals (see BTR-score and Smith//Score); changes the results of these two methods, no others | a pairwise tie in BTR-score and BTR-Borda split evenly |
| 2026-10-25 | Bootstrap draws per condition, from its elections and `--seed` (`analyze.bootstrap_weights`), in `analyze.py` and `compare.py`: both give a condition the same draws, and adding runs leaves the other conditions unchanged; `compare.py` defaults to 1000 replicates and draws a beatpath of clear dominance (`beatpath.png`, `--beat-share`); simulation results unchanged, borderline dominance calls in reports may differ | one random stream across all conditions |
| 2026-10-23 | `run.simulate(..., workers=0)` runs without worker processes; the tests use it, so they also run where new processes start a fresh interpreter (macOS, Windows, Linux from Python 3.14); results unchanged | — |
| 2026-10-23 | Method set `extended` (standard + ESR, DMS, Geometric median, Soft Minimax, Soft Minimax-40, MARS-T) and the reference runs ([reference-runs.md](reference-runs.md)); results unchanged | — |
| 2026-10-23 | Fingerprints: each run records per method a hash of its results in a small fixed test (column `fingerprint`); `--resume` and `compare.py` refuse runs of other code, `analyze.py` notes them; results unchanged | no fingerprints |
| 2026-10-23 | `compare.py`: methods across all runs in a folder (each condition counted once, mixed versions refused unless forced); results unchanged | — |
| 2026-10-22 | New method ESR (expanding support runoff, rated; tested as the add-on BSR-S, results under that name read as ESR) | — |
| 2026-10-21 | Groups: `median-based` (MJ, MAV, MCA, from rated) and `two-round` (TTR, Approval runoff, Approval cond. runoff, Score runoff, from plurality-based, approval-based, hybrid); add-ons can give a group per method. Results unchanged | the earlier groups |
| 2026-10-21 | Dominance and Pareto front: sincerity differences below 1 point count as ties | any clear sincerity difference counted |
| 2026-10-21 | `--utility linear` (−distance / width) and `quadratic` (−(distance / width)²) as alternatives to the exp fall-off; default and results unchanged | exp only |
| 2026-10-20 | Plots and report: the frontier of normalized ballots as a dotted line (new column Un_cands); results unchanged | no line |
| 2026-10-20 | New methods: DMS (rated) and Score runoff (hybrid, a separate runoff; in the standard set, now 29 methods) | 28 methods |
| 2026-10-19 | Standard set: Copeland//Margins-ER removed (28 methods), not a live proposal | 29 methods, with Ranked Robin |
| 2026-10-19 | Renamed: Ranked Robin → Copeland//Margins-ER, Ranked Robin-SR → Copeland//Margins (Equal Vote's "Ranked Robin" now means any Condorcet method; the lab's is Copeland with a margins tiebreak). Results unchanged | use the old names |
| 2026-10-17 | Project renamed Ballot Front Finder Lab (was Voting Pareto Lab); licensed under the GNU GPL, version 3 or later | Voting Pareto Lab |
| 2026-10-17 | `--cluster-salience` removed: `--cluster-salience-exponent 0` gives exactly the electorates of `none` (column `salience` filled in from the exponent) | `--cluster-salience none` or `cluster` |
| 2026-10-17 | Default strategy: every voter adaptive (`--strategic 0`, `--adaptive` left out = everyone not always strategic; with `--converge` 0) | `--strategic 0.5 --adaptive 0` (half the voters always strategic) |
| 2026-10-17 | Honest approval: a personal threshold between the voter's mean and favourite (uniform draw per voter), for Approval, Approval runoff, Approval cond. runoff, MDDA, MAMPO (and Inverse-Approval) | every candidate above the voter's mean approved (ranked for MDDA, MAMPO) |
| 2026-10-17 | Options of the clusters model renamed `--cluster-*` (`--cluster-count`, `--cluster-spread`, `--cluster-zipf`, `--cluster-salience`, `--cluster-salience-exponent`, `--cluster-fixed-variance`); file columns unchanged | `--clusters`, `--spread`, `--zipf`, `--salience`, `--salience-exponent`, `--fixed-variance` |
| 2026-10-17 | Manipulation test, two-sided: the defence tries every exaggeration of its ballot type and counters with the most effective one | the first exaggeration only |
| 2026-10-17 | Report: Low and Low\* (winner below the median candidate) | — |
| 2026-10-17 | Manipulation test (`--manipulation`), new diagnostic | — |
| 2026-10-17 | Default selection: the standard set (29 methods); new method MAV | every method except references and inverse |
| 2026-10-17 | Ranked pairs no longer hangs with many tied margins (11+ candidates); BTR methods and 321 use at most 720 tie orders (from 7 tied candidates) | all tie orders |
| 2026-10-16 | Laziness per voter: each voter's cut uniform between 0 and `--lazy` | one cut (`--lazy`) for everyone |
| 2026-10-16 | Adaptive voters (`--adaptive`): strategic per round by their stake in the leading race; the default `--adaptive 0` leaves results unchanged | — |
| 2026-10-16 | Default electorate `noise` (strength 5, 6 dimensions, width 0.7); `clusters` defaults 20 clusters, no Zipf, fixed total variance, salience per cluster with exponent 0.5 (see Electorate models, and the new defaults) | `clusters`, 10 Zipf dimensions, 5 clusters, width 1.0 |
| 2026-10-13 | Truncated rankings: strategic voters may rank fewer candidates (unranked tied last), and `--lazy` makes honest ballots lazy ([model.md](model.md), Truncation and laziness). Changes the strategic results of every ranking method, also at `--lazy 0`, and the rules for unranked candidates in IRV, Coombs, Borda, Dowdall, Supplementary vote and ER-IRV | full rankings |
| 2026-10-12 | Sincerity counts only reversed preferences, weighted by utility difference and scaled by the most dishonest ballot the format allows; fidelity is new ([analysis.md](analysis.md), Sincerity and fidelity). Both blank for TRASH and the Inverse-\* methods. Since sincerity is a front axis, fronts may change | sincerity divided by all pairs (a high floor for plurality and approval); not comparable |
| 2026-10-12 | COWPEA Lottery removed (a proportional method, judged unfairly by single-winner measures, and the slowest); group condorcet-hybrid renamed condorcet-based | report COWPEA Lottery and the old group name |
| 2026-10-11 | Every run or sweep writes its own folder (`output/<out>/`, analysis in `output/<out>/analysis/`); `analyze.py x` reads it | `--out x` wrote `output/x-elections.csv`, given to `analyze.py` as a file |
| 2026-10-11 | 10 polls per round (`--samples`). A test with 1 to 30 showed systematic shifts only at 3 or fewer; from 5 up the results moved by a few points without a trend (strategy noise that more elections average out). 10 take 37% of the time of 30 | 30 polls |
| 2026-10-10 | 3 poll rounds by default; every row records the simulator version (1.0.0 until publication) | round 1 unless they say otherwise |
| 2026-10-09 | Dominance judged on VSE\*, WME\* and sincerity; honest VSE and WME are context | other axes |
| 2026-10-09 | Removed: the Dowdall variants (Black-, BTR-, Baldwin-, Nanson-Dowdall; Dowdall stays) and the contingent vote (Contingent vote, Contingent-ER): the Dowdall test had answered its question (see the current picture of 2026-10-09), and the contingent vote only served to separate the automated from the real runoff | report them |
| 2026-10-07 | Renamed: Top-two runoff → TTR, Approval runoff if needed → Approval cond. runoff (ACR), Minimax-wv → MMWV | use the new names |
| 2026-10-06 | Axes VSE and WME, both 0 = random ballot and 100 = best candidate ([analysis.md](analysis.md), Axes). Within a condition a fixed rescaling of each axis, so order, dominance and rank stability are unaffected | PE (0 = utility-proportional lottery) and WM (worst margin in % of voters) |
| earlier | Plain dimensions before Zipf; CE as an axis before WM | |

## BTR-score and Smith//Score (2026-10-26)

BTR-score was introduced as a variant of Smith//Score that "gives exactly the same results unless there is a cycle of 4 or more candidates" ([Four levels of voting methods](https://hiveism.substack.com/p/four-levels-of-voting-methods)). In the reference runs of 1.0.0 the two were close with honest ballots (base: VSE 87.0 against 87.1) but drifted apart under strategy (VSE\* 86.7 against 87.3).

**The argument holds for strict pairwise results.**
- With a Condorcet winner, both elect it.
- A candidate outside the Smith set loses every match against a member of it, so it never changes BTR-score's outcome.
- In a cycle of three, the two lower-scored members meet first, and the top-scored member beats whichever of them survives. So BTR-score elects the highest-scored member of the cycle, as Smith//Score does.
- Only Smith sets of four or more can differ.

**The difference came from pairwise ties.** Smith//Score includes tied candidates in the Smith set and elects the highest score total among them, so a tie is decided by score in effect. BTR-score split a pairwise tie evenly between the two candidates. On a 0–5 ballot many voters give two candidates the same score, so exact ties are not rare.

Checked on 2000 base electorates, honest ballots:

| | 0–5 | 0–99 |
|---|---|---|
| electorates with at least one pairwise tie | 11.7% | 7.1% |
| electorates with a Smith set of 4 or more | 0.9% | 0.8% |
| winners differ, tie split evenly | 1.6% (28 with a pairwise tie, 3 with a Smith set of 4+) | 1.4% (24 and 3) |
| winners differ, tie to the higher score | 0.2% (1 and 3) | — |

Strategic score ballots exaggerate, which makes equal ratings and so pairwise ties more common; that is a likely reason why the methods drifted apart more under strategy than with honest ballots (not measured).

**Decision:** BTR-score now gives a pairwise tie to the candidate higher in the score order, the order it already uses. This completes its definition, which left ties open, and makes the claim above hold in practice. BTR-Borda, the same procedure on Borda points, follows the same rule; with complete rankings and an odd number of voters its pairwise ties are rare, but laziness (truncated rankings) makes them possible. The change alters the results of both methods, so the version is now 1.1.0 (README, Versions); results of 1.0.0 for these two methods are not comparable with later ones. BTR-score's fingerprint changes with the rule; BTR-Borda's does not, since the fingerprint's six test elections contain no pairwise tie for it, so for BTR-Borda only the version number tells old and new runs apart. The regression check showed one changed value for BTR-score and none for BTR-Borda.

## ESR, expanding support runoff (2026-10-22)

Tested as the add-on BSR-S, a variant of the Buck-set runoff (BSR) of the previous section: at every depth k (the top k score levels) the candidates with the most score points from those levels form the set, instead of the most voters; then the stretched score runoff among them. At the full depth the points are the score totals, so the Score winner is always in the set. Built in as ESR; the voter-counting BSR stays a test add-on.

Runs (defaults otherwise; VSE\* / WME\*):

| Run | ESR (BSR-S) | BSR | Score | Soft Minimax-40 | Geometric median |
|---|---|---|---|---|---|
| adaptive voters, 300 elections | 90.0 / 93.3 | 88.0 / 95.1 | 89.6 / 93.8 | — | — |
| full strategy (`--strategic 1 --adaptive 0`), 7000 elections, seed 4 | 86.1 / 95.3 | 85.3 / 95.9 | 84.8 / 93.6 | 85.8 / 96.3 | 88.1 / 92.2 |
| exp utility, all methods, 200 elections | 88.0 / 93.1 | 86.5 / 95.3 | 89.1 / 93.2 | 87.5 / 93.3 | 89.7 / 89.2 |
| linear utility, all methods, 200 elections | 97.7 / 98.2 | 96.9 / 97.5 | 96.5 / 93.4 | 97.7 / 97.0 | 96.3 / 94.2 |
| quadratic utility, all methods, 200 elections | 97.0 / 91.8 | 96.9 / 93.2 | 96.4 / 83.6 | 97.1 / 87.9 | 94.4 / 90.7 |

- **With adaptive voters** ESR sits next to Score: with points, the Score winner tops most depths, the set is often that one candidate.
- **Under full strategy** it is undominated, between Geometric median and Soft Minimax-40, and ahead of Score on both axes (+1.3 VSE\*, +1.7 WME\*): strategy costs it 3.4 VSE against Score's 5.0. Inflated ballots are stretched afresh over the set, which undoes much of the exaggeration. Here it also beats the voter-counting BSR, which twelve methods clearly dominate.
- **Across utility shapes** it stays near the top under linear and quadratic utility as well, where the pure sums (Score, DMS, Geometric median) lose WME\*: one of the robust methods, with the Soft Minimax family, MARS, BTR-score and Score runoff.
- The name: "expanding" for the depths (Bucklin's expanding approval), "support" for the points counted at each.

## Depth maximin; how far score ballots can go (2026-10-20)

Settings: the defaults (adaptive voters, 0–5), single runs of 300 to 3000 elections and one sweep over poll rounds 1–5 and max score 5, 9, 19, 99 (600 elections each, the same electorates in all 20 conditions). The methods were tried as test add-ons first.

**Depth maximin.** A score ballot read as levels from the top: at depth k, the top k levels. DMS gives each candidate, per depth, the points from those levels divided by the points of the levels counted, and elects the candidate whose lowest value over the depths is highest. Eleven variants were tried around it: approvals instead of points (DMA), values relative to the field, from depth 2 on, flatter and steeper level weights, a soft minimum, elimination with re-stretched ballots (DMS-I), the worst winning-votes defeat subtracted (DMS-WV), best plus worst depth value (DMS-BW), and a Buck-set runoff (BSR: the candidates with the most voters in the top k levels at some depth, then a stretched score runoff). None was better than DMS: the variants move along the trade-off or fall inside it (DMS-BW: the best value brings back a plurality-like count). BSR lands next to ARA (more WME\*, a little less VSE\*). A Buck-set of one need not hold the Condorcet winner; a random search found 2 such profiles in about 1300.

| 3000 elections, seed 2, 3 poll rounds | VSE\* | WME\* | VSE | WME |
|---|---|---|---|---|
| Geometric median | 90.3 | 90.4 | 90.7 | 89.9 |
| DMS | 90.0 | 91.3 | 90.7 | 89.8 |
| Score | 89.8 | 92.1 | 90.3 | 92.8 |
| ARA | 89.2 | 94.6 | 89.6 | 94.5 |
| BSR | 87.5 | 95.0 | 86.6 | 95.7 |
| STAR | 87.8 | 95.5 | 88.1 | 95.6 |

- **DMS sits with Score and Geometric median.** In early runs of 300 elections it led on VSE\* by about 2 points; with 3000 the lead is gone (+0.2, not clear), and Score is clearly ahead on WME\*. Over the 20 sweep conditions DMS averages 90.8 / 90.4, Geometric median 90.6 / 90.6, Score 90.0 / 91.7; DMS against Geometric median ranges from −0.8 to +1.2 VSE\* by condition, within the noise of 600 elections. Geometric median is dominated by Score and DMS in the 3000-election run (WME\* and sincerity).
- **Poll rounds** change none of them systematically. **Max score** moves ARA along the trade-off (about 89 / 95 at 0–5 to 87.5 / 97.5 at 0–99) and leaves the rated methods where they are.
- **Why the rated methods stop near VSE 90.** A scratch check on 200 default elections, honest ballots: the lab's frontier rule (each candidate's total utility against their worst margin, every weighting) fed with ballots instead of utilities, and judged on the true utilities. Ballots stretched to each voter's range but not rounded reach at most VSE 89.8 (at WME 91.5); rounded to 0–100, 0–10 and 0–5 the best VSE is 89.8, 89.9 and 89.2. The best WME falls with rounding: 100, 99.9, 99.4, 95.3 (equal scores carry no pairwise preference). So the VSE loss of score ballots is the stretching (a ballot cannot show how much one voter cares compared with another), not the rounding; the WME loss at 0–5 is the rounding. Honest Score is the VSE end of the 0–5 curve by construction, Smith//Score sits at its WME end, and the curve passes about one point outside both. The strategic VSE\* of Score, DMS and Geometric median is within a point of their honest VSE, which is already at that limit: on VSE there is little left to gain for any count of score ballots.
- **Kept:** DMS, as a built-in rated method (not in the standard set). Added with it: Score runoff, the two highest score totals in a separate runoff election decided by sincere preferences, in the standard set as the reference between Score and STAR.

## Laziness and approval against real ballots (2026-10-17)

**Question:** which `--lazy` to use as a default, and does the model's honest ballot resemble real ones? IRV ballots were set aside: their truncation is partly strategic (lower preferences rarely count), and the strategic voters already model that. Data: PrefLib ([preflib.org](https://preflib.org)) sets 00001 (Irish STV, 2002: Dublin North, Dublin West, Meath), 00002 (Debian, Condorcet/Schulze: leader elections 2002–2012 with "None of the above", logo vote) and 00007 (ERS: 87 elections of organisations, mostly STV, 3–29 candidates); and a collection of approval elections and experiments (St. Louis 2021, Fargo 2022, the French presidential experiments 2002–2017, the US 2012 and 2016 experiments, Mesel 2008, Konstanz 2009). Model: defaults (noise model, 99 voters), sincere ballots, 40–200 electorates per candidate count.

**Ranked ballots** (share ranking only one candidate / ranking all, i.e. at least C − 1; "None of the above" not counted as a candidate):

| Data | Candidates | One | All | Model value that fits |
|---|---|---|---|---|
| Debian leader 2002, 2012 | 3 | 5%, 11% | 95%, 89% | about 0.1 (12% / 88%) |
| Debian leader 2003, 2010 | 4 | 3–4% | 87–89% | below 0.1 (0.1: 2% / 75%) |
| Debian leader 2005–2007 | 6–8 | 4–5% | 70–77% | below 0.1 (C = 8, 0.1: 0% / 56%) |
| ERS, median | 3–4 / 5–6 / 7–9 / 10–14 / 15–30 | 23% / 16% / 6% / 4% / 5% | 66% / 36% / 28% / 20% / 9% | 0.3–0.5 / 0.5 / 0.3 / 0.2–0.3 / — |
| Irish STV | 9 / 12 / 14 | 6% / 4% / 5% | 16% / 10% / 5% | about 0.5 (C = 12: 4% / 10%) |

- Two regimes: in Debian's elections (engaged voters, few candidates, Schulze) almost everyone ranks fully, which fits a cut near 0.1 or below; in public and organisational STV elections with many candidates ballots are much shorter, fitting 0.2–0.5. Truncation grows with the number of candidates in the data, as in the model (the cut is in utility, so more candidates fall below it).
- Every data set has a few percent of bullet votes even where nearly everyone else ranks fully. One uniform cut cannot produce both; a mixture (a share of bullet voters plus a small cut) would. Not built: one parameter suffices for a robustness check.
- **Decision:** the default stays `--lazy 0`; 0.1 and 0.5 are documented as robustness values ([model.md](model.md), Truncation and laziness).

**Approval ballots** (approvals per ballot): real voters approve 1.5–3 candidates almost regardless of the field (St. Louis mayor, 4 candidates: 1.56; Konstanz, 6: 1.94; Fargo, 7: 1.58; Mesel, 8: 1.86; the French experiments with 10–12 candidates: 2.2–2.6, with 16: 3.15; Hérouville 2017, 11 candidates: 23% per candidate, about 2.5). The model's honest approval ballot (every candidate above the voter's mean utility) approves about 0.4 per candidate: 1.76 with 4, 2.53 with 6, 2.92 with 7, 4.03 with 10, 4.79 with 12, 6.38 with 16. Laziness hardly changes this (the mean lies above most cuts; even at `--lazy 1` 2.25 with 7 and 4.10 with 16), and neither do candidates spread wider (`--cand-dispersion` 3 with 11 candidates: 3.60 approvals, but the top candidate approved by 95%, real 37–65%). Real voters seem to approve the few candidates near their favourite, not everyone above their average; part of that may be strategy (approving relative to the front-runners). Learned ballots do not close the gap (approvals per ballot after 3 poll rounds, 200 elections, defaults otherwise; Approval, the two approval runoffs within 0.1 of each other): honest → learned with `--strategic 0.5` / `--strategic 0 --adaptive 1` / `--strategic 1`: 4 candidates 1.74 → 1.61 / 1.62 / 1.48; 6: 2.52 → 2.30 / 2.29 / 2.06; 7: 2.91 → 2.64 / 2.64 / 2.36; 11: 4.44 → 3.98 / 4.01 / 3.54; 16: 6.36 → 5.62 / 5.80 / 4.9. Strategy removes about a tenth of the approvals (a fifth if everyone is strategic); with many candidates the model still approves 1.5–2 times as many as real voters, so the honest rule is the main cause. Other honest rules, approvals per ballot for 4 / 6 / 7 / 11 / 16 candidates (30 electorates each): mean (current) 1.72 / 2.64 / 2.98 / 4.40 / 6.38; midpoint of the voter's range 1.65 / 2.41 / 2.58 / 3.34 / 4.37; 70% of the range 1.36 / 1.68 / 1.84 / 2.10 / 2.56; mean + 1 standard deviation 1.00 / 1.20 / 1.41 / 1.92 / 2.67; mean of the better half 1.00 / 1.35 / 1.81 / 2.47 / 3.24. Not changed yet: it affects Approval, Approval runoff and Approval cond. runoff.

**Fitting the whole curve, not only the count** (scratch analysis; the concern: which candidates get the few approvals carries information, a cut fitted to the count may fix the wrong end; real fields are pre-selected, so a few candidates matter and the rest hardly). Target: the sorted approval shares of the 11 real approval elections with at least 4 candidates and complete shares (France 2007 lists 8 of 12 and is left out). Model: defaults, 199 voters, 80 electorates per candidate count; mean over elections of the error in the count, the root mean square difference of the sorted shares, and the error in the top candidate's share:

| Honest approval rule | Count | RMS of shares | Top share |
|---|---|---|---|
| a: above the voter's mean (current) | +1.30 | 0.169 | +15% |
| c: as a, candidates spread wider (`--cand-dispersion 2`) | +1.05 | 0.177 | +28% |
| b: personal threshold centred at N/(N+k), N = sum of stretched utilities ("effective number"), k = 1 / 2 / 4 | −0.37 / +0.37 / +1.55 | 0.093 / 0.104 / 0.227 | −11% / 0% / +12% |
| b': as b with k = 1 and the number of candidates for N | −0.67 | 0.115 | −16% |
| d: b (k = 1) with `--cand-dispersion 2` | −0.47 | 0.100 | +5% |
| e: vse-sim's BulletyApproval60 (60% bullet votes, 40% rule a); with 40 / 50 / 70% bullet votes | −0.04; +0.39 / +0.18 / −0.28 | 0.079; 0.088 / 0.079 / 0.089 | −9%; −2% / −5% / −12% |
| **g: personal threshold uniform between the voter's mean and their favourite** | −0.11 | **0.076** | −8% |

- g fits best and has no fitted constant: a candidate's chance of approval rises linearly from 0 at the voter's mean to 1 at the favourite, a gradient instead of a hard cut. Its count grows like the data (4 candidates 1.28, 9: 1.96, 11: 2.21, 16: 2.93; real 1.24–1.56, 1.73, 2.41, 3.00). Every good variant approves the top candidate somewhat less often than real voters (Fargo 65%, the US 2012 experiment 90%).
- A selected field alone (c) does not help: the mean rule approves the near candidates however far the rest are.

**Effect on the results** (scratch run: the rule g for the approval and approval-runoff ballots only, each voter's draw fixed by their utilities so it is the same for every method; defaults, 200 elections, seed 1; Score, STAR, Smith//Score, IRV and Plurality unchanged, as they should be). Honest VSE / WME → learned VSE\* / WME\*, current rule vs g:

| Method | `--strategic 0.5`, current | g | `--strategic 0 --adaptive 1`, current | g |
|---|---|---|---|---|
| Approval | 77.7 / 83.1 → 81.4 / 89.5 | 86.5 / 84.1 → 84.5 / 89.1 | 77.7 / 83.1 → 85.0 / 90.1 | 86.5 / 84.1 → 86.9 / 84.7 |
| Approval cond. runoff | 77.6 / 83.7 → 82.7 / 89.3 | 86.3 / 90.8 → 84.6 / 95.4 | 77.6 / 83.7 → 83.3 / 91.0 | 86.3 / 90.8 → 86.4 / 94.2 |
| Approval runoff | 86.2 / 96.0 → 85.1 / 98.2 | 85.7 / 92.1 → 84.0 / 94.0 | 86.2 / 96.0 → 86.1 / 96.9 | 85.7 / 92.1 → 86.7 / 94.1 |
| (Score, for comparison) | 89.2 / 92.5 → 86.9 / 94.3 | same | 89.2 / 92.5 → 89.1 / 93.2 | same |

- Honest Approval gains 9 points of VSE: approving fewer, but by degree of liking, makes the expected approval ballot a score ballot above the voter's mean (randomly rounded), so Approval moves toward Score. This matches vse-sim's remark that its "ideal" (mean) approval was not ideal, and that BulletyApproval60 did not hurt.
- Strategy then adds little to Approval's VSE, and under adaptive voters its WME\* falls (90.1 → 84.7): strategy no longer repairs an overly generous honest ballot. Approval cond. runoff improves on every axis; Approval runoff loses some WME (fewer approvals make the top two less consensual).
- The approval methods' results depend strongly on the honest rule: worth stating whichever rule is used.

**Adopted** ([model.md](model.md), Honest approval): rule g replaces the mean rule, with each voter's draw from its own random stream (the same for every method; a strategic group's reference ballot uses the mean draw), for the approval ballot, the round-1 approvals of the approval runoffs and the truncated ranking of MDDA and MAMPO (ranked = approved by their definition). ARA, ARA-Top and the other score-based methods are not affected: their ballots are scores (stretched to the voter's range), and reading scores above the middle as approval is part of the method's count, not a voter's decision. Regression: only Approval, Approval runoff, Approval cond. runoff, MDDA, MAMPO and Inverse-Approval changed.

Check (defaults, 200 elections, seed 1; earlier rule = every draw 0, which is the threshold at the mean). Honest VSE / WME → learned VSE\* / WME\*:

| Method | `--strategic 0.5`, earlier | new | `--strategic 0 --adaptive 1`, earlier | new |
|---|---|---|---|---|
| Approval | 77.7 / 83.1 → 81.4 / 89.5 | 86.9 / 85.7 → 86.0 / 87.8 | 77.7 / 83.1 → 85.0 / 90.1 | 86.9 / 85.7 → 88.5 / 88.4 |
| Approval cond. runoff | 77.6 / 83.7 → 82.7 / 89.3 | 87.5 / 92.2 → 85.6 / 96.3 | 77.6 / 83.7 → 83.3 / 91.0 | 87.5 / 92.2 → 86.6 / 92.1 |
| Approval runoff | 86.2 / 96.0 → 85.1 / 98.2 | 86.1 / 93.4 → 84.7 / 95.0 | 86.2 / 96.0 → 86.1 / 96.9 | 86.1 / 93.4 → 84.6 / 93.5 |
| MDDA | 83.5 / 91.6 → 83.0 / 94.0 | 86.9 / 85.7 → 86.9 / 89.5 | 83.5 / 91.6 → 84.7 / 92.3 | 86.9 / 85.7 → 86.0 / 88.5 |
| MAMPO | 87.1 / 90.9 → 87.8 / 94.1 | 86.5 / 85.2 → 87.5 / 89.8 | 87.1 / 90.9 → 87.6 / 91.6 | 86.5 / 85.2 → 85.2 / 88.2 |
| ARA, Score, STAR | unchanged | | | |

- With short honest rankings MDDA's majority disqualification almost never acts (a majority must rank a rival above a candidate, and unranked candidates are only tied last): honest MDDA equals Approval (86.9 / 85.7 in both), and MDDA and MAMPO lose 5–6 points of honest WME. Their Condorcet-like protection depended on the long rankings the mean rule produced. Real voters rank most candidates when asked to rank (Debian, above), so whether a voter facing "ranked = approved" behaves like an approval voter (few) or a ranking voter (many) is open. A variant with an explicit approval cutoff (a full ranking plus approval marks, as the approval runoffs' ballot) would separate the two; not built.

**Approval map** (`show_electorates.py --sav`, Alós-Ferrer & Granić's space from approval ballots) now uses the same honest approval. Cumulative inertia explained by 1 / 2 / 3 / 4 dimensions (median of 100 electorates of 1000 voters, seed 1, defaults: 6 dimensions, width 0.7, noise strength 5; no `--sav-noise`), mean rule → new rule, and the share of elections with a pair of candidates nobody approves together:

| Model | 6 candidates | 8 candidates |
|---|---|---|
| noise | 73 / 94 / 100 / 100 → 67 / 93 / 100 / 100 (57% → 82%) | 62 / 85 / 94 / 99 → 56 / 83 / 95 / 99 (78% → 96%) |
| clusters | 68 / 93 / 100 / 100 → 60 / 90 / 100 / 100 (34% → 67%) | 57 / 81 / 94 / 99 → 51 / 79 / 93 / 99 (31% → 79%) |
| ks | 77 / 95 / 100 / 100 → 72 / 96 / 100 / 100 (88% → 95%) | 69 / 89 / 96 / 99 → 64 / 89 / 97 / 100 (98% → 100%) |
| field | 56 / 86 / 100 / 100 → 52 / 83 / 99 / 100 (3% → 14%) | 48 / 75 / 91 / 99 → 45 / 72 / 89 / 99 (12% → 42%) |
| real (Alós-Ferrer & Granić 2015) | Konstanz: 54 / 91 / 98 / 100 | Messel: 35 / 58 / 73 / 84 |

The first dimension explains 3–8 points less in every model, a step toward the real maps (which stay flatter, especially Messel). With fewer approvals, pairs nobody approves together become common (given half a voter, see [model.md](model.md)), which makes the map less reliable; real ballots of about 1000 voters rarely leave such gaps, a hint that real approvals are noisier than the model's (`--sav-noise` blurs them).

`--sav-noise` (blurring each utility by a normal error of s × the voter's spread before approving), noise model, 60 electorates of 1000 voters, 6 / 8 candidates: s = 0: 67 / 93 / 100 / 100 (gaps in 88% of elections), 57 / 84 / 95 / 99 (97%); 0.25: about the same; 0.5: 72 / 92 / 100 / 100 (47%), 58 / 83 / 94 / 98 (55%); 1: 65 / 88 / 98 / 100 (0%), 51 / 76 / 88 / 95 (0%). **Decision:** default 1 (was 0): no gaps left to patch, and with 8 candidates closer to Messel (35 / 58 / 73 / 84). Only the approval map is affected; the methods' honest approval has no noise beyond the personal threshold.

**Score ballots** (a first check, 2026-10-17): the honest score ballot (the voter's range stretched over the scale, rounded) against the in-situ evaluation ballots of Voter Autrement 2017 in Hérouville-Saint-Clair (11 candidates; per candidate the share of grades in the lower half; from Baujard et al.'s slides, the only figures at hand). Model: defaults, 60 electorates of 499 voters, 11 candidates. Sorted shares of lower-half grades per candidate:

| Ballot | Model | Real | Mean, model / real |
|---|---|---|---|
| 6 levels (0..5), lower half 0–2 | 43 49 55 60 66 71 76 79 83 88 93 | 48 49 50 76 81 87 88 88 94 95 95 | 69% / 77% |
| 4 levels, lower half | as above | 44 50 52 80 80 87 87 87 97 97 97 | 69% / 78% |
| approval (new rule), not approved | 59 65 70 74 78 81 84 87 89 92 95 | 46 48 51 75 81 85 87 88 95 95 95 | 80% / 77% |

- The honest score ballot is somewhat more generous than real grades (8 points), far less than the earlier approval rule was; not changed.
- The larger difference is the field: in the data three serious candidates are graded low by about half the voters and the other eight by 75–97%, a cliff; the model's candidates spread evenly. This is the selection of candidates (a few likely winners, many who hardly matter), a property of the candidate model, not of the ballots. Open for a later version; one town, one non-binding experiment.

## Manipulation and tails with adaptive voters; the defence fixed (2026-10-17)

**Run**: defaults (noise model, 99 voters, 6 candidates, 3 poll rounds, 10 polls, aggressive share 0.5), standard set, 600 elections, seed 1, adaptive voters only:

```
python run.py --strategic 0 --adaptive 1 --elections 600 --manipulation
```

Condorcet winner in 97.8% of the electorates (587 of 600).

- **Main result.** Score has the highest VSE\* (90.0 / WME\* 92.7), followed by ARA, MAV, 321 and MJ (88–89). The Condorcet and minimax methods have WME\* 99.0–99.5 at VSE\* 83.4–84.4. Strategy helps Approval (honest 79.4 → 86.0) and Plurality (69.7 → 80.3) most, as in the earlier adaptive runs. Undominated: Score, Smith//Score, Soft Minimax-D, MARS, MMPO, MMWV. 40–51% of the adaptive voters are strategic per method (most under Plurality, least under Approval cond. runoff).
- **Low and Low\*** (share of elections with a winner below the median candidate, honest → learned): Plurality 11.7% → 7.2%, Supplementary vote 7.3% → 5.2%, TTR 4.8% → 6.0%, the Condorcet methods 4.5–5.0%, Score 3.5% → 3.3%, MJ 2.3%, Approval 1.8% and Approval cond. runoff 2.0–2.3%, the lowest. The order differs from VSE: Approval's honest VSE is low (79.4) but it almost never elects a poor candidate; its losses are many small ones. At 600 elections the intervals are about ±1.5 points, so only the extremes are clear.
- **One-sided manipulation** (share of elections manipulable; VSE when the worst successful manipulation is recorded; honest VSE in the main table): the runoffs protect best: TTR 23% (VSE 73.6), IRV-3 37% (65.2), Approval runoff 40% (71.2), then BTR-score and Smith//Score 48–49% (63.2, 61.1), the Condorcet and minimax methods 58–59% (53–55), IRV 53% (49.3). Most exposed: Borda 92% (−10.0), Score 88% (6.3), Approval 85% (28.6), Plurality 73% (−10.0). Broadly Huang's picture: Score is the most manipulable rated method, Smith//Score and STAR hold up better than Score and Approval. Not used for dominance; a worst case with informed coalitions, beside the poll rounds' typical strategy.
- **Two-sided, and a flaw it showed.** Every Condorcet method with strict rankings (Ranked pairs, Schulze, Kemeny-Young, Minimax, Copeland, BTR-IRV) was manipulable against the defence in exactly 2.2% of the elections: the 13 of 600 without a Condorcet winner. With one, T's supporters always restore T. Ranked Robin, MMWV and MMPO, however, reached 20–28%. They use rankings with ties, whose first exaggeration (the one the defence used) ties T with every candidate the defender likes more than T instead of ranking T first and X last: a weak defence. The defence used the first exaggeration of each ballot type, which is not always the strongest.

**Fix** ([model.md](model.md), Manipulation test): the defence now tries every exaggeration of its ballot type and answers with the one that leaves the coalition the least gain; a manipulation stands only if it pays against every answer. A first attempt, the answer best for the defenders' own total utility, was not kept: it raised Schulze to 18% (60 elections), because the defenders sometimes prefer an answer that elects a third candidate the coalition likes too. Check (60 elections, one poll round, 11 methods; only two-sided changes): Ranked Robin, MMWV and MMPO now 1.7% like Schulze and Minimax; Borda 25% → 12%; IRV 27% → 20%; Score, Approval, STAR and Plurality unchanged (their first exaggeration was already the strongest). Regression: only manip2 and U_manip2 changed.

**Rerun with the fix** (the same command; every other column identical to the first run, as expected). Two-sided (share manipulable against the defence; VSE when the worst standing manipulation is recorded):

- Every Condorcet method, now including Ranked Robin, MMWV and MMPO (before 20–28%), is manipulable in 2.2% of the elections, only those without a Condorcet winner; VSE 82.2–83.5, about their honest VSE (83.8–84.3).
- Then Approval runoff 8.3% (one-sided 39.7%; VSE 81.8), BTR-score 11.0%, Smith//Score 11.7%, Soft Minimax-D 12.3%, STAR 12.8%, MARS 13.0%, Borda 14.5% (first run 52.3%), ARA 14.7%, IRV and ER-IRV 16.3% (ER-IRV first run 27.0%), Score 17.7%, IRV-3 18.5%, TTR 18.7%.
- Most exposed: MJ 21.3%, 321 21.7%, MAV 24.2%, Approval cond. runoff 25.8%, Approval 26.5%, Supplementary vote 27.2%, Plurality 37.2% (VSE 55.7).
- One- and two-sided rank the methods differently. TTR resists best one-sided (23.2%) but the defence hardly helps (18.7%): plurality ballots give T's supporters little to answer with. The Condorcet methods are average one-sided (58%) but a defended Condorcet winner cannot be beaten. Score falls from 87.7% to 17.7% and from VSE 6.3 to 80.6: its worst cases rely on T's supporters staying sincere.
- Changed by the fix beyond the check: 321 23.2% → 21.7%, Score 18.3% → 17.7%; the other shares unchanged (some two-sided VSE moved by up to 0.8, e.g. IRV 74.3 → 73.5).

## Tails: how often the winner is poor (2026-10-17)

Huang's votesim also reports the worst 5% of elections; the lab's efficiencies are ratios of sums and hide how often a method elects a poor candidate. Added to the report and summary.csv: **Low** and **Low\*** ([analysis.md](analysis.md), Axes and other measures), the share of elections whose winner lies below the median candidate by total utility, with honest and with learned ballots. A share of poor outcomes instead of a percentile of per-election VSE, because one election's VSE is unstable (small denominators). Computed in analyze.py from the stored candidate totals, so it needs no rerun and works for older runs too; not used for dominance.

## Manipulation test (2026-10-17)

John Huang's votesim ([summary report](http://votesim.usa4r.org/summary-report.html), 5 February 2021; [code](https://github.com/johnh865/election_sim)) measures strategy as a worst case: for each candidate X other than the honest winner T, the voters preferring X to T use a fixed tactic (bury, compromise, truncate, bullet, …) with perfect information; one-sided, the successful tactic worst for the electorate is recorded; two-sided, T's supporters answer with a bullet vote. His results (average of honest, one- and two-sided VSE): Smith//Score 0.870, STAR 0.858, Smith-minimax 0.850, Ranked pairs 0.849, MJ 0.812, Approval 0.772, Score 0.762, IRV 0.725, top-two 0.683, Plurality 0.515, broadly the order found here. The lab's poll rounds already have front-runners, blocs and both sides; what was missing is the worst case, which Huang argues matters because successful strategy invites more strategy.

**Built** ([model.md](model.md), Manipulation test): `--manipulation`, a diagnostic beside the poll rounds. Coalitions for each X against T, the naive exaggerations of each ballot type as tactics; one-sided: share of elections with a successful coalition and the VSE of the worst successful manipulations; two-sided: the same when T's supporters answer (in this first version with their ballot type's first exaggeration for T; see the section above), counting only manipulations that still pay (votesim keeps backfires too and uses a bullet vote). Regression now runs it in every condition. A first look (20 elections, standard set, defaults; indicative only): Score and Plurality fall furthest one-sided (VSE 2.6 and −10.5), Condorcet methods are manipulable one-sided in about half the elections but almost never against the defence.

## Standard set and MAV (2026-10-17)

**Standard set** ([methods.md](methods.md), Groups): a selection across the groups, now the default of run.py and sweep.py, so a standard run concentrates on the methods that matter most and the others stay available by name (`--methods all` for every one). The selection: the methods in Quinn's vse-sim (Plurality, IRV, Borda, Approval, Score, STAR, MJ, MAV, 321, Ranked pairs, Schulze), common proposals (TTR, ER-IRV, IRV-3, Approval runoff, Approval cond. runoff, Ranked Robin, Minimax, MMWV, MMPO) and the strongest found here (MARS, ARA, BTR-score, Smith//Score, Soft Minimax-D; of the Soft Minimax variants the one undominated most often in the recent runs), plus Copeland (the simplest Condorcet method), Kemeny-Young (the textbook reference), Supplementary vote (in real use) and BTR-IRV (an IRV–Condorcet hybrid): 29 methods. Geometric median and Condorcet-random-drop are left out as idealised reference cases rather than practical methods. Default runs before this change ran every method except the references and the inverse methods.

**MAV**, Majority Approval Voting (graded Bucklin), new in the group rated, on score ballots (0..`--max-score`): counted grade by grade from the top; at the first grade where some candidate has a majority, the one with a majority and the most votes at the grade above wins. Rules from electowiki; vse-sim's version uses five grades and a fractional median score instead (its `candScore`), which is close but not identical.

## Hang with 11 or more candidates fixed (2026-10-17)

A sweep over 3 to 12 candidates got stuck from 11 candidates on. The cause was an integer overflow in Ranked pairs: equal margins are locked in every order (averaged), up to 240 orders, and the number of orders was counted with `np.prod`, which overflows from 21 tied pairs and can turn negative; a negative count passed the limit, and the method then tried to list every order of 21 or more pairs. With 11 candidates there are 55 pairs, and small polls give few distinct margins, so such groups occur. Now counted with exact integers. Same kind of limit added to the tie orders of the BTR methods and 321 (`orders_by`): beyond 720 orders (from 7 tied candidates), 720 fixed random orders instead of all. Regression unchanged (at most 5 candidates there). Run time per election with all methods, one measurement: 9 candidates 50 s, 10: 70 s, 11: 130 s, 12: 180 s.

The part of that sweep that finished (3 to 10 candidates, 200 elections each) was too noisy to be useful; a candidate sweep needs more elections per condition, which is affordable with fewer methods (`--methods`).

## Laziness per voter (2026-10-16)

**Why:** with one cut for everyone (next section, Adaptive voters: With laziness), laziness is either negligible or makes every ballot format equivalent; real ballots are heterogeneous (some voters bullet-vote, many rank fully, some stop after a few).

**Change** ([model.md](model.md), Truncation and laziness): each voter's cut is drawn per election uniformly between 0 and `--lazy` (own random stream, the same for every method), so `--lazy` is now the largest cut. A strategic group's reference ballot uses its members' mean cut. Lazy runs from before this change used one cut for everyone and cannot be told apart in the files; they should not be combined with newer ones. Regression: only the two lazy conditions changed.

**Runs**: defaults (noise model, 200 elections, seed 1, 3 poll rounds, aggressive share 0.5) with `--strategic 0 --adaptive 1`, at `--lazy 0`, `0.5` and `1` (commands of the form `python run.py --strategic 0 --adaptive 1 --lazy 0.5 --out adaptive-test-0-100-0-lazynew50`). Honest → strategic (VSE / WME):

| Method | no laziness | `--lazy 0.5` | `--lazy 1` |
|---|---|---|---|
| Minimax | 82.8 / 100.0 → 81.4 / 99.0 | 87.6 / 93.9 → 86.7 / 95.8 | 88.4 / 89.6 → 88.0 / 91.5 |
| Copeland | 83.0 / 99.8 → 83.0 / 97.7 | 87.6 / 93.8 → 85.9 / 94.2 | 88.2 / 89.5 → 86.4 / 91.2 |
| IRV | 85.6 / 88.1 → 84.1 / 91.4 | 86.5 / 87.2 → 87.2 / 90.0 | 87.9 / 85.8 → 86.5 / 89.7 |
| Borda | 78.9 / 95.0 → 82.1 / 94.1 | 87.7 / 94.1 → 86.5 / 92.2 | 88.2 / 89.1 → 89.3 / 91.2 |
| Approval | 77.7 / 83.1 → 85.0 / 90.1 | 81.9 / 86.2 → 85.5 / 90.3 | 88.5 / 87.3 → 88.1 / 89.7 |
| Score | 89.2 / 92.5 → 89.1 / 93.2 | 90.7 / 90.6 → 90.4 / 90.5 | 87.9 / 85.7 → 88.7 / 90.2 |
| STAR | 86.6 / 94.6 → 85.3 / 94.7 | 87.7 / 91.3 → 87.9 / 92.6 | 89.0 / 88.0 → 88.5 / 90.0 |
| MARS | 87.6 / 95.0 → 85.2 / 95.2 | 89.5 / 91.9 → 88.3 / 92.9 | 88.6 / 87.3 → 89.6 / 92.8 |
| Smith//Score | 86.6 / 95.3 → 86.1 / 95.8 | 88.0 / 92.3 → 87.5 / 92.7 | 89.1 / 88.3 → 90.2 / 91.3 |
| Plurality | 69.2 / 71.8 → 77.8 / 80.7 | unchanged | unchanged |

- **At 0.5 the methods stay distinguishable**, unlike with one cut for everyone. Rank correlation with no laziness: WME\* 0.91 (noise reference 0.93), sincerity 0.96 (0.95), VSE\* 0.61 (0.74), so a moderate change on VSE\* only. The spread of honest WME across methods narrows from a standard deviation of 5.7 to 4.0.
- **Truncation moves the Condorcet methods toward the utility winner:** honest VSE up, WME down (Minimax 82.8 / 100 → 87.6 / 93.9). Approval's honest result improves (77.7 / 83.1 → 81.9 / 86.2: fewer approvals near the mean cut), its strategic result barely moves. Plurality and TTR are unchanged, as their marks name a single candidate anyway.
- **At 1** (the average voter cuts at the middle of their range) the formats come close to each other again: honest WME of almost every ranked and rated method 86–90; rank correlation VSE\* 0.18, WME\* 0.62, sincerity 0.95. The effect of a uniform cut, now only at the extreme end.
- Undominated: no laziness Score, MD-MARS, Smith//Score, Baldwin-ER; 0.5 Score, Soft Minimax-40, MARS-T, MARS, MMWVxScore, ARA; 1 MARS, Score, Soft Minimax-D. Score is undominated in all three, MARS in both lazy conditions.

## Adaptive voters (2026-10-16)

**Why.** In the reruns for a new base, a fixed strategic share combined badly with laziness: lazy sincere voters do not separate the front-runners, while the strategic share optimises its ballots to the limit over the rounds. This is starkest for approval and score, whose ballots leave the most room. A fixed share is a useful first approximation (cheap, and comparable with vse-sim), but realistically voters strategise when they have a reason.

**First attempt: dissatisfaction (not kept).** Adaptive voters turned strategic, and stayed so, once a poll's expected result fell below a threshold of their range from least to most liked candidate (`--threshold`). Run: defaults (noise model, 200 elections, 3 poll rounds) with `--strategic 0.5` against `--strategic 0 --adaptive 1 --threshold 0.5`. The strategy became one-sided: the voters losing in the polls attacked and the content supporters of the likely winner never defended. Score's VSE\* fell from 86.9 to 75.2 (honest 89.2), Approval's from 81.4 to 76.2 (honest 77.7), Borda's from 81.7 to 75.2, while Minimax (82.1 → 82.8), Schulze (82.1 → 82.0) and STAR (86.9 → 86.1) barely moved; burial under the ranked methods roughly halved and push-over doubled (Minimax 13.0% → 5.5% and 3.4% → 6.8%). The VSE\* order changed (rank correlation 0.37, noise reference 0.68), WME\* (0.86) and sincerity (0.95) did not. Under every method 40–47% of the adaptive voters turned strategic: the trigger picked the voters far from the likely winner, whatever the method. The same kind of artefact as the early naive burial: moves without the counter-moves.

**Kept: the stake in the leading race** ([model.md](model.md), Partial strategy). Each round, an adaptive voter's stake is |u(A) − u(B)| for the round's two leaders, rescaled to 0–1; the voter is strategic in that round if it exceeds their personal threshold (uniform on 0–1, fixed per election), so the share strategic equals the stake. Decided every round anew, so voters can turn sincere again. Symmetric: in a close race both leaders' supporters have a stake. No new parameter beyond `--adaptive`; `--threshold` was removed. A distribution of personal thresholds with a parameter (a sigmoid around a midpoint) was considered and left out until results ask for it.

**First check** (40 elections, 9 methods, otherwise defaults; indicative only): with `--strategic 0 --adaptive 1`, Score's VSE\* is 88.8 (share model 89.1, honest 91.8) and Approval's 91.9 (share model 90.3); the collapse is gone. 43–50% of the adaptive voters strategic per method.

**Comparison with the share model**: defaults (noise model, 200 elections, seed 1, 3 poll rounds, aggressive share 0.5), `--strategic 0.5` against `--strategic 0 --adaptive 1` (commands of this form):

```
python run.py --out default
python run.py --strategic 0 --adaptive 1 --out adaptive-test-0-100-0
```

- About as much strategy as the share model: 41.5–51.6% of the adaptive voters strategic per method (most under Plurality, least under Borda-ER).
- The method order holds: rank correlation with the share model VSE\* 0.82 (noise reference 0.68), WME\* 0.89 (0.93), sincerity 0.97 (0.98). The dissatisfaction version had reordered VSE\* (0.37).
- Methods stay closer to their honest WME: the mean shift from honest to strategic is 2.4 points under the share model and 1.4 with stakes (VSE: 1.4 under both). Above all the rated and hybrid methods, which had gained WME from the random strategic half exaggerating with little at stake: 321 +8.1 → +1.0, MCA +7.5 → +3.0, MJ +3.7 → −2.5, Score VSE −2.3 → −0.1 and WME +1.8 → +0.7.
- Approval-based methods gain more (on average +2.9 VSE, +3.3 WME): Approval from 77.7 / 83.1 honest to 85.0 / 90.1 (share model 81.4 / 89.5), consistent with the leader rule (both leaders' supporters adjust their approvals around the leading pair; Laslier 2009). Borda keeps its WME (Borda-ER −3.1 → +0.1). Plurality gains least strongly (+8.6 VSE, share model +11.8).
- Condorcet and minimax methods lose a little more VSE (−0.5 to −1.4 on average; BTR-Borda and ER-IRV −2.7). Push-over nearly disappears under the ranked methods (Minimax 3.4% → 0.8%): it pays mostly for voters with little at stake in the leading race, who now stay sincere.
- Undominated: Score, MD-MARS, Baldwin-ER (share model: Soft Minimax-L, Soft Minimax-D, MARS, IC-MARS, Soft Minimax, MMPO). One seed of 200 elections: provisional.

**With laziness** (as above, `--strategic 0 --adaptive 1` with `--lazy 0.5` and `--lazy 0.1`, laziness then still one cut for everyone):

- At 0.5 the methods collapsed already with honest ballots: honest VSE 85–90 for almost every method (Plurality 69.2, DSC 75.2), the Condorcet methods' honest WME 100 → about 86 and their VSE 82.8 → 88.7; rank correlation with the base for honest VSE 0.24 and WME 0.25 (noise references 0.89 and 0.97). With everyone cutting at the middle of their range, every format says roughly "these candidates are acceptable", so the methods behave like approval with a cut at the midpoint. A property of the laziness model, not of the adaptive voters.
- At 0.1 only ranked ballots changed (rated and approval results identical to no laziness, honest and strategic): the Condorcet methods' honest WME 100 → 97.2, VSE 82.8 → 84.5, strategic WME\* 98.3 (without laziness 99.0). Order close to the base: VSE\* 0.80 (0.68), WME\* 0.85 (0.94), sincerity 0.96 (0.98).

## Electorate models, and the new defaults (2026-10-14 to 2026-10-16)

**Outcome.** Four electorate models were compared; the fractal-noise model (`noise`) became the default, at strength 5, 6 dimensions and width 0.7 ([model.md](model.md), Electorates: Defaults, and why). It needs the fewest assumptions and gives electorates between the extremes of the others: `clusters` is clumpy (with 5 clusters, islands), `ks` uniform and low-dimensional, `field` a featureless blob. At the defaults a Condorcet winner exists in about 97% of elections, matching real ranked elections (96–98% in Myers' CIVS data with 50 to 100 or more voters), and electing them loses about 15% of VSE, so the methods separate. The `clusters` model's own defaults became its best variant below (20 clusters, no Zipf, fixed total variance, salience per cluster with exponent 0.5).

Across all models and settings tried, the earlier finding held: **the methods' order on WME\* and sincerity hardly depends on the electorate; their order on VSE\* does**, and mainly through how much electing the Condorcet winner costs.

**All four models side by side** (electorates only, no methods: `show_electorates.py`, 300 electorates per model, seed 1, 3000 voters, 6 candidates; noise strength 4, 6 dimensions, width 0.7; the other models at their parameters of the time: clusters 5 clusters with Zipf, field saturation 1.5; approval map from 100 electorates of 1000 voters):

```
python show_electorates.py --model clusters noise ks field --noise-strength 4 --dims 6 --width 0.7 --sav --voters 3000
```

| Model | CW | CW = utility winner | VSE lost electing CW | fav ÷ least | first two components | approval map, cumulative inertia 1/2/3/4 dims |
|---|---|---|---|---|---|---|
| clusters | 98.0% | 65.3% | 16.7% | 8.6 | 89% | 76 / 96 / 100 / 100 |
| noise | 99.0% | 77.4% | 11.8% | 6.6 | 60% | 70 / 93 / 100 / 100 |
| ks | 99.7% | 78.9% | 5.8% | 6.0 | 100% | 77 / 96 / 100 / 100 |
| field | 99.7% | 93.3% | 0.4% | 4.2 | 37% | 56 / 86 / 100 / 100 |

Real approval data with 6 candidates (Konstanz, Alós-Ferrer & Granić 2015): 54 / 91 / 98 / 100. With more parties or candidates the real maps are flatter (Messel, 9 parties: 37 / 54 / 66 / 76; Konstanz, 11 parties: 24 / 45 / 57 / 67); no model has been compared at those sizes with method runs.

### The noise model (2026-10-15/16)

Voters are dense where a fractal noise is high ([model.md](model.md), Electorates). Two sweeps over its strength, otherwise at the defaults of the time (3 poll rounds, 10 polls of 49 voters, poll error 2, search 10, aggressive share 0.5, strategic share 0.5, scores 0..5, 99 voters, 6 candidates, seed 1):

```
python sweep.py --model noise --noise-strength 3 4 6 --out noise          # of this form; 10 dimensions, width 1.0, 200 elections
python sweep.py --model noise --width 0.7 --dims 6 --elections 600 --noise-strength 5 6 7 --out noise2
```

| Strength (dims, width) | CW | CW = utility winner | VSE lost electing CW | fav ÷ least | reachable corner (VSE at best WME / WME at best VSE) | undominated |
|---|---|---|---|---|---|---|
| 3 (10, 1.0) | 96.5% | 87.6% | 2.5% | 2.6 | 97.3 / 95.5 | ARA-Top, IC-MARS, MARS, Approval |
| 4 (10, 1.0) | 99.5% | 83.9% | 5.0% | 2.8 | 95.0 / 91.1 | Soft Minimax-40, Soft Minimax-D, MARS, Approval runoff |
| 6 (10, 1.0) | 98.0% | 69.9% | 15.5% | 3.5 | 83.6 / 79.7 | Soft Minimax-40, MARS, Soft Minimax-D |
| 5 (6, 0.7) | 97.8% | 70.7% | 15.3% | 7.9 | 84.3 / 77.0 | Geometric median, Soft Minimax, MMPO, Minimax-ER, Schulze-ER |
| 6 (6, 0.7) | 96.0% | 69.1% | 16.6% | 9.7 | 81.4 / 77.8 | Geometric median, MARS, Soft Minimax-40, Smith//Score, Soft Minimax-D, MD-MARS, Baldwin-ER |
| 7 (6, 0.7) | 97.0% | 67.9% | 17.2% | 12.2 | 80.8 / 75.2 | ARA, MARS-T, MARS, Soft Minimax-D, Approval runoff, Baldwin-ER, Minimax-ER |

- **Strength 3 and 4 are easy electorates**: nearly every method is close to the best possible result. At strength 6 with 10 dimensions and width 1.0, the reachable region opens up and the methods separate.
- **From strength 5 to 7 (6 dimensions, width 0.7) the model is on a plateau.** The cost of electing the Condorcet winner rises only from 15.3% to 17.2%, and the method order stays within its noise: rank correlation with strength 5 for VSE\* 0.90 and 0.89 (noise reference 0.85), WME\* 0.96 and 0.95 (0.98), sincerity 0.98 (0.99). Only the steepness (favourite ÷ least liked) keeps growing. So the exact strength matters little in this range; 5 is the lowest value clearly past the easy regime.
- **In the first sweep, strength 3 → 6 changed the VSE\* order** (rank correlation 0.23, noise reference 0.50), but not the WME\* order (0.81; 0.90) or the sincerity order (0.94; 0.97): as in all model comparisons.
- Unlike `clusters`, where a costlier Condorcet winner came with a rarer one, here the Condorcet winner stays frequent at every strength and only its cost grows.

Selected methods at strength 5 (6 dimensions, width 0.7, 600 elections; each value's own VSE\* interval is about ±2.5, but paired differences are much more precise):

| Method | VSE\* | WME\* | Sincerity | VSE | WME |
|---|---|---|---|---|---|
| Geometric median | 90.6 | 92.1 | 99.9 | 91.3 | 90.4 |
| ARA | 88.2 | 96.3 | 99.9 | 89.9 | 94.4 |
| MARS | 87.8 | 95.9 | 100.0 | 88.7 | 95.7 |
| Soft Minimax-40 | 87.8 | 95.8 | 99.9 | 89.2 | 94.1 |
| Score | 87.5 | 94.9 | 100.0 | 90.8 | 92.6 |
| Soft Minimax-D | 87.5 | 96.5 | 99.9 | 87.9 | 96.1 |
| STAR | 87.5 | 96.2 | 99.9 | 88.2 | 95.7 |
| Smith//Score | 86.9 | 97.0 | 99.9 | 87.5 | 96.4 |
| Soft Minimax | 86.7 | 97.6 | 99.9 | 87.0 | 96.6 |
| BTR-score | 86.3 | 97.5 | 99.9 | 87.1 | 96.4 |
| IRV | 85.0 | 94.1 | 96.2 | 85.9 | 88.7 |
| Approval runoff | 84.5 | 98.1 | 99.8 | 85.2 | 96.2 |
| Ranked pairs | 83.9 | 99.6 | 96.5 | 84.3 | 100.0 |
| Minimax | 83.5 | 99.5 | 96.4 | 84.3 | 100.0 |
| Plurality | 80.0 | 81.4 | 96.2 | 69.7 | 70.8 |

**A sampling artefact, found and fixed before the runs.** The first prototype drew voters by weighting a fixed pool of Gaussian points by exp(strength × noise). Above strength 2 the weights concentrate on a few dozen points, so about half the voters were copies of each other, and the costly electorates it showed were mostly that. The model samples by raising the strength in steps with resampling and local moves instead ([model.md](model.md), Electorates), so all people are distinct. Prototype numbers from before this fix are not valid above strength 2.

### Variants of the clusters model (2026-10-15)

| Run | settings | CW | CW = utility winner | VSE lost electing CW | fav ÷ least | reachable corner | undominated |
|---|---|---|---|---|---|---|---|
| standard-clusters | 10 dims Zipf, 5 clusters, width 1.0, 200 elections | 96.0% | 66.1% | 12.5% | 4.3 | 85.3–86.8 / 77.0 | Soft Minimax, Smith//Score, Condorcet-random-drop, Schulze-ER, Minimax-ER |
| clust-20 | the same with 20 clusters | 98.0% | 78.6% | 6.0% | 4.8 | 93.3 / 90.8 | MARS-T, MMWVxScore, Soft Minimax-L, Soft Minimax |
| new-defaults | 6 dims, no Zipf, 20 clusters, width 1.5, 200 elections | 96.0% | 80.7% | 7.7% | 5.7 | 90.3–90.8 / 87.0 | MARS-T, Soft Minimax-40, IC-MARS, Smith//Score, Soft Minimax-L |
| sal-05 | 10 dims, no Zipf, fixed total variance, 20 clusters, salience per cluster, exponent 0.5, width 1.0, 600 elections | 93.5% | 77.5% | 8.9% | 3.8 | 88.3–88.7 / 86.1 | IC-MARS, Score, MARS, MARS-T, Soft Minimax-D, MMWVxScore, Smith//Score, Schulze-ER |

- **5 → 20 clusters**: the five-cluster electorates are islands with gaps between them; with 20 clusters electing the Condorcet winner costs half as much. Rank correlation (noise reference in brackets): VSE\* 0.72 (0.86), WME\* 0.93 (0.95), sincerity 0.97 (0.97).
- **Equal dimensions are sensitive to their number.** new-defaults against clust-20 (10 dims with Zipf, width 1.0) changed the VSE\* order (0.31; noise reference 0.60), not WME\* (0.86; 0.82) or sincerity (0.95; 0.98). In electorate-only probes, without Zipf the cost of electing the Condorcet winner grew with the number of dimensions even at fixed total variance. Global Zipf scaling roughly doubled the variance decay that clustering alone already produces (the principal components of clustered electorates fall off steeply even with equal dimensions), so the decay does not need to be imposed. Choosing dims and width to hit one target looked like over-fitting, which led to the salience option.
- **Salience per cluster** (each cluster weighs the issues in its own order) made the cost of electing the Condorcet winner stable against the number of dimensions, but made Condorcet winners rarer: 87.8% with exponent 1 and 10 dimensions in an electorate probe, 93.5% with exponent 0.5 (sal-05). sal-05 looked reasonable, with a balanced front; it is now the default of `clusters`.

### What did not work (2026-10-15)

- **The field model** (opinions as scale-free fields over a social space, saturation 1.5, 10 issues, width 0.7, 600 elections, run field-07): CW 95.8%, CW = utility winner 85.4%, electing it loses 3.9%, fav ÷ least 3.0, reachable corner 95.5–95.8 / 95.5; undominated IC-MARS, Soft Minimax-40, MMWVxScore, Soft Minimax, Smith//Score, Score. It hardly separates the methods. The issues are independent fields, so no ideology (issues that go together) forms: a featureless blob. It stays in the lab as an experimental model.
- **Dynamic opinion models, as prototypes outside the lab** (a 40 × 40 grid of agents with 10 issues, elections as snapshots of 99 voters and 6 candidates, width 0.7, 60 snapshots each after 2000–3000 burn-in sweeps; not in the code). Bounded confidence with noise (Deffuant with noise, Pineda, Toral & Hernández-García 2009): agents closer than ε move toward each other on one issue, and with probability m an issue is reset at random. It switches sharply from disorder (ε ≤ 0.15: no groups, Condorcet winner in 96–100%, cost 2–3%) to one consensus group with a halo of reset agents (ε ≥ 0.2), with no lasting structure of several groups; electorates are easy (cost 0–3%). Adding repulsion between distant agents (Jager & Amblard) pushed 82–100% of opinions to the edges of the issue space, with independent issues: random corners of a cube, Condorcet winner in only 52–82%, cost 9–62%. Moving on all issues at once instead of one did not couple the issues. Both versions end at one of two poles, consensus or independent yes/no issues; real electorates lie between.

### Open idea: recombination for the field model

A static way to let structure emerge from merging and diverging, without agents (proposed 2026-10-16, not tried): split top-down for about ten levels, as now, but give each new faction two parents instead of one (their average plus a fresh deviation scaled to its level). Then repeatedly drop the oldest level and add a new one at the bottom, so lower levels move up and the structure turns over completely two or three times, which removes the initial conditions. Two points decide the outcome:

- The balance of averaging and deviation (averaging two parents halves the spread, the deviation adds it back) is the temperature between uniformity and noise.
- Which parents merge. With random pairs the turnover mixes everything into a Gaussian blob, the field model's present state. Structure can only survive if merges mostly join nearby factions (as assortative mating in population genetics); how strongly nearby factions are preferred would be the knob between blob and islands.

### Clusters vs Quinn's hierarchical model (2026-10-14)

**Settings.** Two runs at the defaults, differing only in the electorate model: 200 elections, seed 1, 99 voters, 6 candidates, width 1.0, 3 poll rounds, 10 polls of 49 voters, poll error 2, search 10, aggressive share 0.5, strategic share 0.5, scores 0..5, default methods. `clusters`: 10 dimensions (Zipf), 5 clusters, spread 0.5. `ks`: faction alpha 1.5, cut 0.2 ([model.md](model.md), Electorates). Analysed together (settings as the report states them; the commands were of the form):

```
python run.py --model ks --out standard-ks
python run.py --out standard-clusters
python analyze.py standard-ks standard-clusters --out cluster-ks
```

**Every method gains VSE under ks, Condorcet methods most.** VSE\* (learned ballots), clusters → ks:

| Method | VSE\* | WME\* |
|---|---|---|
| Ranked pairs | 84.7 → 96.4 | 99.7 → 98.2 |
| Minimax | 85.3 → 95.2 | 99.1 → 97.8 |
| Smith//Score | 89.8 → 95.5 | 96.8 → 97.9 |
| BTR-score | 89.9 → 95.6 | 96.4 → 97.4 |
| Score | 89.7 → 94.1 | 89.2 → 91.8 |
| Soft Minimax | 90.1 → 94.6 | 97.0 → 97.2 |
| Soft Minimax-L | 92.7 → 94.9 | 93.2 → 93.4 |
| Reciprocal score | 91.3 → 96.4 | 87.5 → 91.8 |
| IRV | 85.5 → 93.3 | 86.5 → 87.2 |
| Plurality | 71.8 → 78.2 | 67.6 → 66.1 |

So the picture is sheared: the methods whose edge was trading a little WME for more VSE (the Soft Minimax family, score hybrids) lose that edge, because under ks the two measures hardly pull apart. The reachable corner moves from about VSE 86 / WME 77 (clusters: VSE at the best WME / WME at the best VSE) to about 95 / 91 (ks). With 200 elections VSE\* intervals are about ±2 points, and under ks the top fifteen or so methods lie within 2 points on VSE\*: their order there is mostly not a finding.

**Fronts.** ks: Pareto front Ranked pairs, Reciprocal score, MARS, MARS-T, Black, BTR-score, Smith//Score, Schulze-ER, Nanson, Soft Minimax-D, SICT, Soft Minimax, IC-MARS, Score; undominated BTR-score, Smith//Score. clusters: undominated Soft Minimax-40, Soft Minimax, Smith//Score, Condorcet-random-drop, Schulze-ER, Minimax-ER. Smith//Score is undominated under both models.

**Rank stability** (Spearman, ks vs clusters; noise reference ks vs itself in brackets): VSE 0.48 (0.55), WME 0.96 (0.98), VSE\* 0.13 (0.65), WME\* 0.91 (0.87), sincerity 0.94 (0.98). The methods' order on WME\* and sincerity hardly depends on the model; their order on VSE\* does.

**Electorates** (from the reports): Condorcet winner in 98.5% (ks) and 96.0% (clusters); it also has the highest total utility in 79.2% and 66.1% of those; electing it loses 4.7% and 12.5% of VSE; favourite ÷ least liked 3.4 and 4.3 (median voter). Under ks the Condorcet winner is nearly the utility winner, so Condorcet methods get high VSE almost for free.

**Why: a first probe of the electorates alone** (no methods; 1000 electorates per variant, 600 for the clusters and width variants, seed 1, defaults otherwise). VSE lost by electing the Condorcet winner (ratio of sums, as VSE), and favourite ÷ least liked:

| Variant | CW = utility winner | VSE lost electing CW | fav ÷ least |
|---|---|---|---|
| clusters | 68.6% | 13.6% | 4.4 |
| clusters, 3 dimensions | 73.4% | 9.3% | 4.6 |
| clusters, 20 clusters | 77.3% | 7.7% | 4.9 |
| clusters, spread 1.0 | 79.4% | 6.6% | 6.0 |
| clusters, width 2.0 | 76.4% | 7.1% | 2.1 |
| clusters, 1 cluster | 88.9% | 1.8% | 2.1 |
| ks | 78.2% | 4.8% | 3.6 |
| ks, shared factions (`--ks-shared 1`) | 78.1% | 5.2% | 3.6 |
| ks, alpha 0.5 | 81.5% | 3.9% | 2.8 |
| ks, alpha 5 | 77.8% | 4.5% | 4.0 |
| ks, width 0.5 | 70.4% | 11.1% | 12.0 |

- Cross-cutting factions are not the main reason: with the same factions in every bundle, ks loses 5.2% instead of 4.8%. (The hypothesis was that coherent blocs across all issues make the Condorcet and the utility winner part.)
- What does matter: how lumpy the electorate is, i.e. a few well-separated blocs. The clusters model loses less with more clusters, wider spread or a single cluster. Its five clusters with Dirichlet weights (often one dominant) and centres twice as far apart as the spread are lumpier than ks factions, whose positions are only about 1.4 times as far apart as their members' spread.
- And how steep preferences are relative to the distances: ks at width 0.5 loses 11.1%, close to clusters at width 1.0, and clusters at width 2.0 loses 7.1%. So part of the difference between the models is the scale of their distances relative to `--width`, not their structure. A comparison of the two structures at matched steepness (e.g. matched favourite ÷ least liked) would separate the two.
- A rough probe (ks with every person weighting the dimensions alike, by their average weight, instead of by their faction's caring) changed nothing (4.8%).

**Method runs with the widths swapped.** The probe suggested that ks at width 0.5 resembles clusters at width 1, and clusters at width 2 resembles ks at width 1. Two more runs, otherwise as above (`--exclude references` changes nothing: the references and the Inverse-\* methods are left out by default, so all four runs have the same 62 methods):

```
python run.py --model clusters --out standard-clusters-w2 --exclude references --width 2
python run.py --model ks --out standard-ks-w0.5 --exclude references --width 0.5
python analyze.py standard-ks-w0.5 standard-clusters-w2 --out ...
```

| Run | CW = utility winner | VSE lost electing CW | fav ÷ least | reachable corner (VSE at best WME / WME at best VSE) |
|---|---|---|---|---|
| clusters, width 1 | 66.1% | 12.5% | 4.3 | 86 / 77 |
| ks, width 0.5 | 69.5% | 12.1% | 11.7 | 87.5 / 84 |
| ks, width 1 | 79.2% | 4.7% | 3.4 | 95 / 91 |
| clusters, width 2 | 78.1% | 5.7% | 2.1 | 93 / 88.5 |

The electorate measures confirm the probe: by how much electing the Condorcet winner costs, each run matches the other model at the other width (steepness, favourite ÷ least liked, does not match: the ks electorates are steeper at the same cost).

Rank agreement of the methods' VSE\* (Spearman over the 62 methods, computed from the two reports' tables; noise reference of a run against itself: 0.65 for ks width 1, 0.70 for ks width 0.5, from the reports):

| | clusters w1 | clusters w2 | ks w0.5 |
|---|---|---|---|
| ks w1 | 0.13 | 0.45 | 0.50 |
| ks w0.5 | 0.68 | 0.70 | |
| clusters w1 | | 0.79 | |

WME\* and sincerity agree between all four runs (0.89–0.98). On VSE\*, every pair agrees about as well as noise allows, except the pairs with ks at width 1. So the large reshuffle of the first comparison is mostly a property of ks at width 1, not of ks against clusters as such: ks at width 0.5 orders the methods like clusters at either width. Why ks at width 1 stands apart is open; a second seed of it (`--seed 2`) would show whether its order replicates, since a single run can fall below its noise reference.

**A lead, not yet a finding: strategy moves the Condorcet methods' VSE in opposite directions under the two models.** VSE\* − VSE (learned minus honest), averaged over twelve ranked Condorcet methods (Ranked pairs, Schulze, Kemeny-Young, Copeland, Minimax, Black, Baldwin, Nanson, BTR-Borda, Benham, Ranked Robin-SR, BTR-IRV): ks +0.3 (width 1) and +1.0 (width 0.5); clusters −0.5 at both widths. Thirteen rated and hybrid methods lose under both (−0.6 to −2.5). The effects are about a point, at the edge of what 200 elections resolve.

## Sincerity, fidelity and the Condorcet winner in the ballots (2026-10-12)

Runs: `python run.py --elections 600` at the defaults (99 voters, 6 candidates, 3 poll rounds, 10 polls, strategic share 0.5, aggressive share 0.5, seed 1), in two parts with `--methods` / `--exclude` by group: `run-20261012-191423` (condorcet, condorcet-based, minimax-family, hybrid, plurality-based, positional) and `run-20261012-195852` (the rest), analysed together (`python analyze.py run-20261012-191423 run-20261012-195852 --out …`; same seed, so the same elections). All 72 methods. 580 elections have a real Condorcet winner, 20 do not. First run with the current sincerity measure, fidelity and the ballot Condorcet winner ([analysis.md](analysis.md), Sincerity and fidelity; Condorcet winner in the cast ballots). Point estimates; the ballot Condorcet table has no intervals.

**Pareto front** (point estimates on VSE\*, WME\*, sincerity; from the report's rounded values, so near-ties may differ in the full-precision analysis): Geometric median, Soft Minimax-L, IC-MARS, Soft Minimax-D, BTR-score, Soft Minimax, MMWV+Score, Condorcet-random-drop, Baldwin-ER, Minimax, Ranked pairs, Schulze-ER, MMWV, MMPOWV. Two branches: the score-based methods (sincerity 99.8–99.9) trading VSE\* against WME\* from Geometric median (VSE\* 93.1) to MMWV+Score (WME\* 96.4), and the ranking-based Condorcet methods (WME\* 98.2–99.4, sincerity 96.5–97.8), which reach higher WME\* at lower sincerity.

**Undominated** (no method clearly better, beyond the paired 95% interval): Soft Minimax-L, Soft Minimax-D, Baldwin-ER. Each of the other Pareto methods is clearly beaten on at least one axis by a method that it beats, if at all, only within noise.

| Method | VSE\* | WME\* | Sincerity | Fidelity | (honest) |
|---|---|---|---|---|---|
| Geometric median | 93.1 | 88.2 | 99.8 | 87.1 | 94.1 |
| Soft Minimax-L | 92.2 | 94.1 | 99.9 | 86.3 | 94.1 |
| Soft Minimax-D | 90.8 | 95.5 | 99.9 | 86.8 | 94.1 |
| Score | 90.8 | 91.4 | 99.9 | 85.5 | 94.1 |
| MARS | 90.5 | 94.6 | 99.9 | 86.3 | 94.1 |
| STAR | 90.2 | 95.0 | 99.9 | 86.8 | 94.1 |
| INS | 89.9 | 94.7 | 99.8 | 86.9 | 94.1 |
| ER-IRV | 87.9 | 86.3 | 97.1 | 76.7 | 81.8 |
| Approval | 87.7 | 88.5 | 99.9 | 66.1 | 69.2 |
| Approval runoff | 87.5 | 95.8 | 99.6 | 65.9 | 69.2 |
| Baldwin-ER | 86.8 | 98.9 | 97.7 | 77.6 | 81.8 |
| Minimax | 86.4 | 99.0 | 96.5 | 76.9 | 81.8 |
| Schulze-ER | 85.5 | 99.4 | 97.7 | 77.5 | 81.8 |
| IRV | 84.9 | 86.3 | 95.8 | 76.1 | 81.8 |
| Borda | 84.0 | 90.1 | 94.8 | 74.4 | 81.8 |
| TTR | 86.0 | 88.5 | 93.8 | 57.1 | 62.8 |
| Plurality | 72.5 | 66.8 | 96.1 | 58.9 | 62.8 |

Geometric median has the highest VSE\* of all methods (93.1) but is dominated on WME\* and sincerity; Score is dominated by Soft Minimax-L on VSE\* and WME\*.

**Sincerity and fidelity: two kinds of strategy.**
- **Score-based ballots (rated and hybrid groups) hardly reverse a preference** (sincerity 99.5–99.9), but they lose the most fidelity to strategy: from 94.1 honest to 85.5–87.4 (−7 to −9; Score −8.6). Their strategy is exaggeration and levelling.
- **Approval ballots** neither reverse much (99.3–99.9) nor lose much to strategy (69.2 → 66): the format has already levelled what strategy would level.
- **Strict rankings reverse preferences** (Condorcet methods 96.1–96.6, IRV 95.8, Borda 94.8) and lose less fidelity: from 81.8 honest (a ranking spaces candidates evenly) to 76–77 (−4 to −6).
- **Equal-ranking (ER) variants sit in between:** sincerity 97.1–97.8, because burying by ties is not a reversal (burial ~19–21% against ~12% for the strict versions), with about the same fidelity as the strict versions.
- **TTR has the lowest sincerity (93.8):** its only strategy, marking a compromise in round 1, is a large share of what a plurality mark can say. Plurality and TTR lose most to the format itself (honest fidelity 62.8).

**The Condorcet winner in the ballots.** What a study of the cast ballots would see, against the real (sincere) Condorcet winner:

| Method | Same | False | None | Spurious (of 20) | CE\* (ballots) | CE\* (real) | CE (honest) |
|---|---|---|---|---|---|---|---|
| IRV | 90.3% | 8.1% | 1.6% | 45% | 82.7% | 79.3% | 66.8% |
| ER-IRV | 91.7% | 6.0% | 2.2% | 35% | 81.1% | 78.4% | 66.8% |
| Supplementary vote | 89.5% | 9.5% | 1.0% | 60% | 69.5% | 62.9% | 54.1% |
| Borda | 89.8% | 6.9% | 3.3% | 55% | 86.5% | 80.2% | 83.8% |
| Minimax | 97.4% | 1.2% | 1.4% | 60% | 100.0% | 98.2% | 100.0% |
| Schulze | 97.4% | 1.0% | 1.6% | 55% | 100.0% | 98.0% | 100.0% |
| Minimax-ER | 95.5% | 1.6% | 2.9% | 25% | 100.0% | 97.8% | 100.0% |
| STAR | 89.5% | 7.6% | 2.9% | 70% | 98.5% | 89.2% | 85.7% |
| Score | 86.2% | 9.0% | 4.8% | 75% | 89.1% | 80.9% | 77.2% |
| MJ | 84.8% | 11.6% | 3.6% | 75% | 88.5% | 78.8% | 68.3% |
| Random pair runoff | 97.8% | 1.7% | 0.5% | 15% | 49.2% | 49.2% | 45.5% |

Same / False / None: shares of the 580 elections with a real Condorcet winner; Spurious: share of the 20 without one in which the ballots show one (rough: 20 elections). CE (honest): the method's Condorcet efficiency with honest ballots.

- **Strategy hides cycles.** In the elections without a real Condorcet winner the strategic rankings show one in 15–60% of cases (IRV 45%), the honest rankings in none. Ballot data that rarely shows a cycle is therefore weak evidence that the electorates had none.
- **Strategy creates false Condorcet winners,** more under the methods that invite compromise: IRV 8.1%, Supplementary vote 9.5%, Borda 6.9%, against about 1% for Schulze, Minimax, Ranked pairs and Kemeny-Young.
- **Ballot data overstates Condorcet efficiency.** IRV elects the ballots' Condorcet winner in 82.7% of the elections where the ballots have one, but the real one in 79.3%; Condorcet methods elect the ballots' winner by construction (100%; MMPO and SICT 99%+) and the real one in 92.3–98.3%. For IRV, strategy also raises the apparent rate well above the honest one (66.8%): voters compromising towards the Condorcet winner is one way IRV comes to elect it.
- **Rated ballots are poor Condorcet data even when honest.** Rounding and ties on 0–5 scores make the honest ballots show another Condorcet winner in 10% and none in 3.3% of elections, and show one in 75% of the elections without one; three grades (MCA, 321) a false one in 20%. Truncated rankings do the same: MDDA and MAMPO (candidates above the voter's mean ranked, the rest tied) show a false one in 24% and none in 9% honestly, IRV-3 (three ranked) none in 10.9%.
- **Where strategy is weak, the ballots stay right:** Random pair runoff's ballots show the real Condorcet winner in 97.8%. In TRASH and the Inverse-\* methods, where leaving the honest ballot pays most, the ballots show the real one in only 37–65%.
- **Not a model of real IRV data.** In these electorates (6 candidates, 10 dimensions) IRV's honest Condorcet efficiency is only 67%, far below what real ranked-choice elections show. The model says that strategy pushes ballot data towards Condorcet consistency, not how large the effect is in real elections.

## First look: Soft Minimax (2026-10-10)

Run (Claude, small): 100 elections, defaults, poll rounds 1 and 3; Soft Minimax with the clip at 20%, 40% and 60% of the score range, against Score, STAR, IC-MARS, INS, Minimax and MMPO. Point estimates; most differences are ties at this size. VSE\* / WME\* / sincerity at round 3:

| Method | VSE\* | WME\* | Sincerity |
|---|---|---|---|
| Score | 92.3 | 89.5 | 99.9 |
| Soft Minimax-60 (not kept) | 91.6 | 90.3 | 99.8 |
| Soft Minimax-40 | 89.7 | 93.0 | 99.8 |
| STAR | 89.4 | 94.3 | 99.7 |
| Soft Minimax (20%) | 89.1 | 97.2 | 99.8 |
| INS | 87.8 | 95.4 | 99.5 |
| Minimax | 82.4 | 100.0 | 97.0 |
| MMPO | 82.3 | 98.5 | 98.1 |

- **The clip moves the method along the trade-off,** from Minimax-like (20%) to Score-like (60%), as designed.
- **Soft Minimax (20%) reached the front at round 3** (with STAR and Minimax): VSE\* tied with STAR, WME\* about 3 points higher, as sincere as the score methods.
- **On the default 0–5 scale it is Minimax on score ballots:** a 1-point clip turns every integer score difference into a whole vote. So most of its advantage over ranked Minimax (VSE\* 89 against 82, sincerity 99.8 against 97.0) comes from the ballot, not the rule: sincere score ballots leave close candidates level, and strategic voters reverse far fewer preferences on scores (burial 2.9% against 12.1%, compromise 0.3% against 7.6%). With a wider score range the 20% clip makes it a genuinely soft rule.
- Kept: Soft Minimax (20%) and Soft Minimax-40. Needs a full-size run.

## New methods: DSC, DAC, STLR, INS, MAMPO, COWPEA Lottery (2026-10-10)

Run (user): `sweep.py --one-at-a-time --strategic 0.5 1 --aggressive 0.5 0 1`, 400 elections, one poll round, defaults otherwise; the new methods with STAR, Score, Approval, MDDA, MMPO, Plurality and Random pair runoff for reference. INS was then still called "Iterated normalized score". Base condition, VSE\* / WME\* / sincerity, and paired differences:

| Method | VSE\* | WME\* | Sincerity |
|---|---|---|---|
| INS | 90.8 | 95.0 | 99.7 |
| Score | 90.4 | 93.4 | 99.9 |
| STAR | 90.3 | 94.9 | 99.8 |
| MDDA | 90.2 | 91.4 | 99.6 |
| MAMPO | 90.1 | 91.8 | 98.9 |
| Approval | 89.6 | 89.6 | 99.9 |
| DAC | 88.2 | 88.3 | 97.6 |
| DSC | 87.4 | 86.9 | 97.1 |
| STLR | 86.9 | 95.6 | 99.8 |
| MMPO | 86.7 | 99.7 | 97.7 |
| Random pair runoff | 54.1 | 43.9 | 98.3 |
| COWPEA Lottery | 27.2 | 26.3 | 99.6 |

- **INS performs well:** tied with Score and STAR on VSE\* and WME\* in every condition, slightly less sincere than Score. Honestly it trades 3.6 points of VSE for 4.1 of WME against Score and has clearly higher Condorcet efficiency (CE +8, CE\* +4 to +8): renormalizing after each elimination recovers pairwise information that plain Score ballots flatten.
- **STLR loses VSE against STAR** (VSE −4.2 honest, VSE\* −3 to −6 in every condition) and ties on WME and WME\*: levelling the runoff to the preferred finalist favours the more widely preferred over the more strongly liked candidate.
- **MAMPO is tied with MDDA and Approval** on VSE\* and WME\*, a little less sincere; honestly it loses WME (−3.4) and CE (−5.4) against MDDA.
- **DAC is at least as good as DSC,** better under full strategy and with all-aggressive voters (WME\* +2 to +4, CE\* +3 to +6): with strict sincere ballots they are identical, and the difference comes from how strategic voters use equal rankings. Both stay in the middle of the field, with burial shares of 16–39%.
- **COWPEA Lottery is a weak lottery reference** (VSE\* 27, below Random pair runoff's 54), and strategy lowers it further (honest VSE 36).
- The front (dominated by none on VSE\*, WME\*, sincerity) is Score and MMPO in three conditions, plus STAR with only kind voters.

## Candidates unlike the electorate (2026-10-10)

Run (user): `sweep.py --one-at-a-time --cand-dispersion 1 0.25 0.5 0.75 1.5 --cand-shift 0 0.5 1`, 200 elections, one poll round, defaults otherwise (strategic share 0.5, aggressive share 0.5), most methods. Voters are the same in every condition; only the candidates are moved ([model.md](model.md), Electorates). Condorcet efficiency with honest ballots, which keeps its meaning across conditions (Condorcet methods: 100 by construction):

| Method | dispersion 0.25 | 0.5 | 0.75 | 1 (base) | 1.5 | shift 0.5 | shift 1 |
|---|---|---|---|---|---|---|---|
| STAR | 66 | 74 | 75 | 86 | 96 | 90 | 90 |
| MARS | 67 | 72 | 75 | 86 | 94 | 90 | 89 |
| Score | 55 | 64 | 65 | 77 | 88 | 80 | 86 |
| Approval | 52 | 57 | 67 | 78 | 85 | 75 | 67 |
| MJ | 47 | 51 | 60 | 69 | 83 | 73 | 77 |
| Borda | 87 | 83 | 86 | 89 | 89 | 81 | 76 |
| IRV | 33 | 36 | 49 | 65 | 87 | 73 | 83 |
| TTR | 33 | 31 | 46 | 68 | 87 | 75 | 89 |
| Plurality | 22 | 21 | 30 | 45 | 67 | 54 | 69 |
| Coombs | 90 | 94 | 97 | 99 | 99 | 98 | 99 |
| Condorcet winner exists | 82% | 93% | 97% | 96% | 99% | 97% | 99% |
| … with the highest total utility | 51% | 52% | 58% | 66% | 83% | 75% | 79% |

With strategy (VSE\* / WME\*), selected:

| Method | dispersion 0.25 | 0.5 | 1 (base) | 1.5 | shift 1 |
|---|---|---|---|---|---|
| STAR | 86 / 80 | 85 / 85 | 92 / 94 | 95 / 98 | 93 / 96 |
| Score | 87 / 74 | 89 / 84 | 91 / 92 | 94 / 97 | 95 / 95 |
| Approval | 77 / 72 | 80 / 78 | 90 / 89 | 93 / 94 | 87 / 85 |
| IRV | 75 / 47 | 79 / 62 | 87 / 84 | 93 / 94 | 91 / 96 |
| TTR | 72 / 49 | 78 / 63 | 87 / 86 | 93 / 97 | 91 / 98 |
| Minimax | 83 / 91 | 80 / 97 | 87 / 99 | 94 / 100 | 91 / 99 |
| Borda | 76 / 82 | 78 / 88 | 81 / 90 | 87 / 91 | 82 / 84 |

- **Clustered candidates hurt every method that is not a Condorcet method,** not only plurality and IRV as in Merrill (1984): from dispersion 1 to 0.25, Condorcet efficiency falls by about 20 points for STAR, MARS, Score, MJ and Approval and by 23–35 for IRV, TTR and Plurality. The order of the methods stays: hybrids, then rated and approval methods, then the IRV family, then plurality. Borda is nearly unaffected (89 → 87), Coombs almost so (99 → 90).
- **The electorates themselves get harder:** with clustered candidates a Condorcet winner exists less often (82% at 0.25), it is less often also the candidate with the highest total utility (51% against 66%), and choosing between the two costs more VSE (16–19% against 13%). Candidates who differ little give voters flat preferences (favourite ÷ least liked 1.5 against 4.3), so small differences decide, which plurality-type first choices and range-stretched scores both handle badly.
- **Under strategy the plurality-based methods suffer most:** IRV and TTR fall to WME\* 47–49 at dispersion 0.25 (base 84–86); the rated methods hold 72–80 and the Condorcet methods about 90. Minimax's WME\* dips most at 0.25 (91) and recovers at 0.5 (97).
- **Spread-out candidates (1.5) make every method better,** and the differences between methods shrink (CE 84–99 except plurality).
- **A shift of 0.5–1 voter SD makes elections slightly easier** (a Condorcet winner more often, and more often also the utilitarian best) and changes the order little; plurality-type methods gain. Approval loses at shift 1 (CE 78 → 67, VSE\* 90 → 87), where many voters approve the same nearest candidates and ties split the win; Borda loses too (89 → 76), for a reason not yet clear.

## First look: candidates unlike the electorate (2026-10-10)

Run (Claude, small): `sweep.py --one-at-a-time --elections 40 --cand-dispersion 1 0.25 0.5 1.5 --cand-shift 0 0.5 1 2`, ten methods, defaults otherwise (one poll round). 40 elections: point estimates, noisy. Condorcet efficiency with honest ballots (CE), which keeps its meaning across conditions:

| Method | base | dispersion 0.25 | 0.5 | 1.5 | shift 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|
| Plurality | 42 | 29 | 28 | 50 | 62 | 70 | 91 |
| TTR | 55 | 47 | 33 | 82 | 72 | 91 | 98 |
| IRV | 52 | 40 | 40 | 78 | 71 | 88 | 96 |
| Approval | 80 | 41 | 53 | 82 | 72 | 64 | 50 |
| Score | 73 | 50 | 58 | 83 | 75 | 88 | 95 |
| STAR | 80 | 53 | 69 | 95 | 88 | 90 | 98 |
| Borda | 81 | 82 | 83 | 86 | 82 | 82 | 92 |
| Condorcet winner exists | 93% | 85% | 90% | 95% | 100% | 100% | 100% |

- **Clustered candidates (dispersion < 1) hurt as Merrill (1984) found** for plurality, TTR and IRV, and in this model just as much for the rated methods: Approval, Score and STAR lose 20–40 points of CE at dispersion 0.25. Preferences become flat (favourite ÷ least liked 1.5 instead of 4.1), and ballots stretched to each voter's own range (scores, above-mean approval) then exaggerate small differences. Borda is unaffected. Spread-out candidates (1.5) help every method.
- **A shifted field makes elections easier,** not harder: there is always a Condorcet winner, and the candidate nearest the electorate is most voters' favourite, so plurality-type methods find it (CE 91–98 at shift 2). Approval is the exception: nearly everyone approves the same two nearest candidates, and the resulting ties split the win.
- **Caution with VSE and WME across shift values:** when one candidate is nearly everyone's favourite, the random ballot is almost as good as the best candidate, the efficiencies' denominators become small, and every miss looks large (Approval's honest VSE falls to 0 at shift 2 from ties alone). Within one condition the order of methods stays meaningful; across conditions, CE and the raw rates are safer.

## Lock-in: starting from a previous election's result, full strategy (2026-10-10)

Run (user): `--start honest previous --rounds 1 2 3 4 5 6 7 8 --strategic 1 --aggressive 1`, 200 elections, defaults otherwise; Score, Approval, IC-MARS, Minimax, IRV, IRV-3, ER-IRV, TTR, Borda. The first poll shows either the honest ballots or the final ballots of one previous (dummy) election under the same method, its candidates relabelled at random ([model.md](model.md), Starting information). Same elections in both; VSE\* / WME\*:

| Method | start | round 1 | round 2 | round 3 | round 5 | round 8 |
|---|---|---|---|---|---|---|
| TTR | honest | 86 / 84 | 89 / 89 | 89 / 90 | 90 / 93 | 89 / 94 |
| | previous | 40 / 30 | 46 / 34 | 47 / 36 | 47 / 35 | 47 / 35 |
| IRV | honest | 85 / 84 | 85 / 81 | 89 / 84 | 89 / 87 | 89 / 88 |
| | previous | 43 / 33 | 50 / 40 | 54 / 44 | 55 / 44 | 54 / 45 |
| IRV-3 | honest | 88 / 83 | 88 / 84 | 87 / 83 | 87 / 86 | 90 / 90 |
| | previous | 64 / 50 | 72 / 63 | 77 / 68 | 80 / 74 | 80 / 77 |
| ER-IRV | honest | 87 / 85 | 86 / 87 | 88 / 89 | 86 / 88 | 87 / 91 |
| | previous | 75 / 67 | 86 / 82 | 87 / 87 | 88 / 89 | 87 / 91 |
| Approval | honest | 84 / 89 | 88 / 89 | 90 / 89 | 85 / 91 | 90 / 91 |
| | previous | 59 / 44 | 82 / 80 | 84 / 87 | 87 / 86 | 90 / 88 |
| Score | honest | 88 / 92 | 89 / 91 | 91 / 92 | 88 / 90 | 90 / 90 |
| | previous | 82 / 74 | 89 / 81 | 89 / 86 | 88 / 87 | 89 / 90 |
| IC-MARS | honest | 86 / 93 | 89 / 94 | 88 / 93 | 90 / 93 | 90 / 93 |
| | previous | 79 / 72 | 87 / 84 | 86 / 90 | 88 / 91 | 87 / 92 |
| Minimax | honest | 83 / 91 | 83 / 84 | 85 / 91 | 88 / 92 | 85 / 91 |
| | previous | 58 / 44 | 79 / 73 | 79 / 82 | 84 / 88 | 83 / 89 |
| Borda | honest | 50 / 42 | 70 / 65 | 61 / 59 | 68 / 68 | 75 / 70 |
| | previous | 72 / 62 | 63 / 61 | 65 / 60 | 69 / 67 | 67 / 59 |

Second run, same settings (user), more methods; VSE\* / WME\*:

| Method | start | round 1 | round 2 | round 3 | round 5 | round 8 |
|---|---|---|---|---|---|---|
| Plurality | honest | 78 / 69 | 80 / 72 | 84 / 73 | 85 / 74 | 85 / 74 |
| | previous | 34 / 26 | 36 / 28 | 36 / 29 | 37 / 30 | 37 / 30 |
| Coombs | honest | 80 / 90 | 83 / 88 | 83 / 89 | 81 / 91 | 83 / 91 |
| | previous | 69 / 56 | 76 / 79 | 78 / 84 | 80 / 86 | 84 / 89 |
| BTR-IRV | honest | 77 / 84 | 80 / 83 | 82 / 87 | 86 / 86 | 87 / 87 |
| | previous | 75 / 69 | 82 / 79 | 85 / 81 | 84 / 81 | 88 / 88 |
| MJ | honest | 87 / 90 | 88 / 89 | 87 / 91 | 87 / 94 | 85 / 93 |
| | previous | 69 / 54 | 84 / 83 | 88 / 88 | 86 / 89 | 83 / 92 |
| MCA | honest | 82 / 91 | 88 / 94 | 83 / 91 | 89 / 89 | 88 / 89 |
| | previous | 73 / 62 | 81 / 82 | 85 / 86 | 89 / 89 | 89 / 93 |
| SICT | honest | 80 / 84 | 84 / 84 | 89 / 89 | 86 / 88 | 86 / 90 |
| | previous | 68 / 54 | 83 / 84 | 88 / 84 | 86 / 91 | 84 / 90 |
| STAR | honest | 85 / 87 | 90 / 94 | 87 / 93 | 86 / 92 | 89 / 93 |
| | previous | 80 / 70 | 86 / 86 | 88 / 91 | 88 / 90 | 88 / 93 |
| ARA, Score+, MMPO, MMWV, Minimax-ER, Baldwin-ER | previous, round 1 | 78–86 / 69–81, back within 2–4 rounds | | | | |
| Condorcet-random-drop | both | 81–87 / 77–86 in every round, no difference between the starts | | | | |

- **TTR and IRV stay locked in.** TTR freezes from round 3 on at about 47 / 35, IRV at about 54 / 44, against 89 / 90 with an honest start. The outdated front-runners keep the first choices: everyone marks or ranks first the preferred of the two old leaders, which keeps them in front, and no single group gains by moving to a candidate the polls show as hopeless. TTR's honest runoff only picks the better of the two old leaders.
- **Plurality locks in completely** (34 / 26 → 37 / 30 by round 8, against 85 / 74 with an honest start) and freezes from round 3 on, like TTR. Together with TTR and IRV it forms the group of methods that count first choices only.
- **Every method that reads the whole ballot recovers.** BTR-IRV and Coombs eliminate like IRV, but BTR-IRV decides each elimination head to head between the bottom two, and Coombs eliminates by last places: the strategic ballots still rank the other candidates below the old leaders, and those full rankings carry the information back. Both are back by round 3–8. The same holds for the rated, approval and pairwise methods, the grade methods (MJ, MCA) and SICT; several (ARA, Score+, MMPO, MMWV, Minimax-ER, Baldwin-ER) are hardly affected even in round 1.
- **ER-IRV recovers by round 2–3,** the same split as in the Inverse test: equal first ranks let voters support several candidates at once, so no coordination on one new front-runner is needed.
- **Approval, Score, IC-MARS and Minimax recover** within 2–5 rounds (Approval 59 / 44 → 84 / 87 by round 3; Minimax 58 / 44 → 84 / 88 by round 5). Score and IC-MARS start least damaged (round 1: 82 / 74 and 79 / 72).
- **IRV-3 recovers partly** (64 / 50 → 80 / 77), clearly better than full IRV, for a reason not yet clear. A guess: ballots that rank at most 3 candidates and leave the other old front-runner unranked exhaust in the later eliminations, which weakens the old leaders' transfers.
- **Borda** is unstable under all-aggressive voters whatever the start (honest 50–75 VSE\* across rounds), so this test says little about it.
- **The honest start flatters the plurality-based methods.** With honest first polls the popular candidates lead, and TTR and IRV look as good as the rated methods (VSE\* about 89). Start them from a different picture and they keep it: their good results depend on the starting information being right. In an electorate whose opinions have moved since the last election, or in which the media picture is wrong, that is the case that matters.

## First look: starting from a previous election's result (2026-10-10)

Run (Claude, small): `sweep.py --start honest previous --rounds 1 3 8 --elections 30 --methods Plurality IRV ER-IRV TTR Approval Score STAR Minimax Borda`, defaults otherwise (strategic share 0.5, aggressive share 0.5). 30 elections: point estimates only, a few points are noise. VSE\* / WME\*:

| Method | honest start, round 1 | previous, round 1 | honest, round 3 | previous, round 3 | honest, round 8 | previous, round 8 |
|---|---|---|---|---|---|---|
| Plurality | 70 / 54 | 49 / 36 | 63 / 47 | 67 / 51 | 74 / 62 | 66 / 59 |
| IRV | 87 / 73 | 68 / 65 | 90 / 88 | 92 / 85 | 87 / 85 | 90 / 86 |
| TTR | 86 / 85 | 75 / 67 | 88 / 81 | 93 / 85 | 84 / 87 | 87 / 80 |
| ER-IRV | 92 / 73 | 87 / 82 | 87 / 74 | 92 / 85 | 93 / 89 | 95 / 88 |
| Approval | 89 / 91 | 87 / 74 | 92 / 87 | 85 / 82 | 94 / 88 | 94 / 91 |
| Score | 96 / 92 | 93 / 86 | 93 / 91 | 89 / 82 | 90 / 92 | 91 / 90 |
| STAR | 95 / 97 | 95 / 91 | 91 / 92 | 95 / 95 | 91 / 93 | 91 / 96 |
| Minimax | 86 / 97 | 89 / 92 | 88 / 100 | 86 / 99 | 89 / 99 | 89 / 99 |
| Borda | 84 / 90 | 84 / 84 | 83 / 93 | 82 / 83 | 88 / 91 | 84 / 92 |

- **Round 1 pays for the outdated picture,** most under Plurality, IRV and TTR (VSE\* 11–21 points lower), less under ER-IRV and the rated, approval and Condorcet methods.
- **By round 3–8 most methods are back,** within the noise of this run; Plurality possibly stays lower (round 8: 66 / 59 against 74 / 62).
- The lock-in is much weaker than with the Inverse-\* methods. Two reasons: the relabelled front-runners are random candidates, not the most disliked ones, and kind voters start from their sincere ballots and keep them unless the polls show a gain, which pulls the polls toward the real electorate. A run at full size, and with `--aggressive 1` and `--strategic 1`, would show whether a lasting lock-in remains for the plurality-based methods.

## Model test: Inverse-\* methods (2026-10-10)

Run (Claude, small): `sweep.py --strategic 1 --rounds 1 3 8 15 --methods inverse Plurality Approval Minimax --elections 40`, with `--aggressive 0.5` (default) and `--aggressive 1`; defaults otherwise. 40 elections: point estimates only, differences of a few points are noise.

The Inverse-\* methods count like their originals, but their sincere ballot expresses reversed preferences, so honest voting elects nearly the worst candidate (VSE about −200). With full strategy and enough rounds, voters should learn to reverse their ballots and get the original's result back.

VSE\* / WME\* with all strategic voters aggressive (`--aggressive 1`):

| Method | round 1 | round 3 | round 8 | round 15 | original, round 15 |
|---|---|---|---|---|---|
| Inverse-Approval | 5 / −1 | 91 / 75 | 87 / 85 | 81 / 88 | Approval 90 / 85 |
| Inverse-Minimax | −22 / −26 | 76 / 84 | 86 / 83 | 85 / 85 | Minimax 87 / 98 |
| Inverse-Plurality | −46 / −69 | −44 / −67 | −44 / −67 | −44 / −67 | Plurality 88 / 65 |

Second small run (Claude, 20 elections, `--strategic 1 --aggressive 1 --rounds 1 3 8`), the further inverse methods; VSE\* / WME\* (Inverse-ER-IRV and ER-IRV from a separate run with the same settings and elections):

| Method | round 1 | round 3 | round 8 | original, round 8 |
|---|---|---|---|---|
| Inverse-Score | 51 / 43 | 88 / 93 | 91 / 89 | Score 85 / 91 |
| Inverse-Borda | 40 / 11 | 75 / 77 | 69 / 57 | Borda 89 / 89 |
| Inverse-IRV | −29 / −76 | 21 / −29 | 26 / −11 | IRV 87 / 87 |
| Inverse-ER-IRV | 66 / 16 | 92 / 84 | 91 / 84 | ER-IRV 90 / 83 |
| Inverse-TTR | −24 / −78 | −10 / −69 | 5 / −49 | TTR 97 / 93 |

- **Approval and Minimax: the voters find their way back.** Within 3–8 rounds Inverse-Approval and Inverse-Minimax reach about the VSE\* of their originals; Inverse-Minimax stays below Minimax on WME\* (85 against 98 in round 15). Sincerity, measured against the reversed preferences the honest ballot expresses, falls to about 37% and 8%: most ballots are reversed.
- **With half the strategic voters kind** (default `--aggressive 0.5`) the recovery is slower and partial (round 12: Inverse-Approval 73 / 70, Inverse-Minimax 66 / 74). Kind voters return to the honest ballot whenever reversing does not strictly pay in the polls, and in a large group a single change rarely does; here the honest ballot is the harmful one.
- **Score recovers like Approval; Borda partly** (20 elections: noisy, and Borda itself is unstable under all-aggressive voters).
- **Inverse-Plurality does not recover.** Honest voting makes two widely disliked candidates the poll leaders. Voters then do the naive leader-rule thing and name the less bad of the two, which keeps both in front: nobody's favourite is a leader, a switch to it gains nothing in the polls, and the electorate stays at about 72% sincerity in a lesser-evil equilibrium. This is the Duverger trap of plurality with a bad start: the strategy model only finds moves that pay against the current leaders, so it cannot coordinate a jump to a new front-runner. Under the ordinary methods voters start from honest ballots, whose leaders are the popular candidates, so this matters less there; but it shows that the poll rounds model level-k responses to the current polls, not ideal strategy.
- **ER-IRV escapes where IRV does not.** Inverse-ER-IRV is back at ER-IRV's level by round 3 (92 / 84), while Inverse-IRV is still at 26 / −11 in round 8. Same count, but equal rankings let a voter put many candidates first at once: the naive exaggeration ties everyone liked at least as much as the preferred front-runner at the top, which hands first-choice support to all of them together, so no coordination on one new front-runner is needed.
- **IRV and TTR lock in too, less completely.** Both decide by first choices first, so the same trap holds: the disliked front-runners of the honest Inverse ballots keep the first-choice votes. TTR's separate runoff, decided by real preferences, picks the less bad of two bad finalists; IRV's eliminations reshuffle a little more. The split is by how the first stage counts: methods that count every candidate on every ballot (scores, approvals, pairwise counts) let voters move many candidates at once and recover; methods that count first choices need coordination on a new front-runner, which poll responses do not produce.

## Current picture: all methods, poll rounds 5, with nomination (2026-10-09)

Run (user): all 59 methods, 400 elections, 99 voters, 6 candidates, defaults (strategic share 0.5, aggressive share 0.5, poll error 2.0x, max score 5), `--rounds 5 --exits 1 --clones 2`. First run with separate runoffs and shared random streams (section below). Replaces the picture of 2026-10-06 further down, which used convergent learning and other axes. Only one round count, so differences between methods not in the "Dominated by" column are point estimates; the frontier's most balanced point is VSE 95.3 / WME 96.7.

Undominated on all four axes (VSE, WME, VSE\*, WME\*; the analysis since judges dominance on VSE\*, WME\* and sincerity only, see the note at the top): **STAR, ARA, MARS, Condorcet-random-drop, Minimax, Schulze, Ranked pairs.** Beaten on a single axis only:

| Method | VSE | WME | VSE\* | WME\* | beaten by (axis) |
|---|---|---|---|---|---|
| STAR | 92.9 | 94.2 | 89.3 | 95.7 | — |
| ARA | 91.5 | 94.9 | 89.6 | 96.2 | — |
| IC-MARS | 91.0 | 95.0 | 89.9 | 96.0 | STAR (VSE) |
| MARS | 90.3 | 95.9 | 88.5 | 97.3 | — |
| Smith//Score | 89.2 | 96.4 | 88.5 | 97.3 | IC-MARS (VSE) |
| MMWVxScore | 88.5 | 96.7 | 88.0 | 97.5 | MARS (VSE) |
| MMWV+Score | 88.5 | 96.5 | 87.0 | 97.0 | MARS (VSE) |
| Condorcet-random-drop | 86.3 | 99.7 | 88.2 | 98.2 | — |
| Minimax | 86.1 | 99.6 | 85.6 | 99.2 | — |
| Ranked pairs | 86.0 | 99.5 | 85.8 | 99.2 | — |
| Schulze | 86.1 | 99.6 | 85.9 | 99.1 | — |
| MMPO, MMWV, MMPOWV, Kemeny-Young, Schulze-ER, Coombs | 85.5–86.1 | 99.1–99.6 | 85.1–86.6 | 98.4–98.9 | Condorcet-random-drop (VSE\*) |

- **Two groups along the trade-off, as before:** score-based runoffs and hybrids (VSE 88–93, WME 94–97) and Condorcet methods (VSE 85–86, WME 99–100). Honestly the Condorcet methods sit almost on the frontier's WME end (86.5 / 100); the best hybrids are about 2.5 points inside it on both axes (STAR 92.9 / 94.2 against the balanced point 95.3 / 96.7), between the −2 and −4 guide lines.
- **Strategy mostly moves the hybrids along the trade-off rather than inward:** STAR 92.9 / 94.2 → 89.3 / 95.7, ARA 91.5 / 94.9 → 89.6 / 96.2, MARS 90.3 / 95.9 → 88.5 / 97.3, IC-MARS 91.0 / 95.0 → 89.9 / 96.0. VSE falls 1–4 points, WME rises 1–1.5. Score loses VSE (92.8 → 90.5) and gains no WME (92.3 → 92.5).
- **Condorcet methods lose little WME\* under strategy;** they differ in how much. Least loss (WME\* 98.8–99.2): Minimax, Ranked pairs, Schulze, Kemeny-Young, Coombs, MMWV, Nanson. More (94.5–97.3): Benham 94.5, Baldwin-Dowdall 95.8, Black-Dowdall 95.9, BTR-IRV 96.5, Black 97.0, SICT 97.0, BTR-Dowdall 97.2, Copeland 97.3.
- **Condorcet-random-drop gains VSE under strategy** (86.3 → 88.2, WME\* 98.2) and beats the minimax family, Kemeny-Young and Coombs on VSE\*. Its burial share (19%) is high, but the burial does not translate into lost WME.
- **Approval runoff with a separate runoff** is among the leaders under strategy (89.5 / 96.4, from 86.3 / 90.5 honest), close to STAR, ARA and IC-MARS. Honestly it is beaten by the hybrids on VSE and WME. Approval (87.4 / 91.8) and Approval cond. runoff (87.7 / 91.4) gain WME under strategy too, but stay far behind.
- **TTR, Contingent vote and Contingent-ER** end up close together (VSE\* 85.9–86.4, WME\* 91.4–92.1), with the same honest values. Only TTR has no burial, since its runoff cannot be manipulated; compromise and push-over remain (9.5% and 8.6%). IRV and ER-IRV gain WME under strategy (85.7 → 91.3 and 92.4); IRV-3 less (88.6). Supplementary vote (80.5) and MCA (91.9, VSE\* 85.3) trail.
- **Dowdall instead of Borda points does not pay in the Condorcet-type methods.** Baldwin-Dowdall and Nanson-Dowdall are not Condorcet-consistent (CE 96.4 and 92.5 honest; WME 98.1 and 96.2): Baldwin and Nanson elect the Condorcet winner because it always has an above-average Borda score, which Dowdall points do not guarantee. Under strategy the Dowdall versions lose more WME than the originals (Baldwin 97.5 vs Baldwin-Dowdall 95.8, Nanson 98.8 vs 94.3, BTR-Borda 98.3 vs BTR-Dowdall 97.2, Black 97.0 vs Black-Dowdall 95.9) for at most a point of VSE\* (Black-Dowdall 86.9 vs 86.2, Nanson-Dowdall 86.4 vs 85.5; Baldwin-Dowdall is lower, 84.2 vs 85.2). Plain Dowdall gains from strategy (85.9 / 84.1 → 87.5 / 88.3) and still trails almost everything, and nomination hits it hard (22% of honest results changed, 0.78 withdrawals per election, against 10% and 0.04 for Borda). An earlier run of the Borda family alone (400 elections, rounds 1–5, nomination on, before the random-stream fix) showed the same pattern in every round. The idea from the poll-rounds section, that Borda-based methods should use Dowdall points, is not supported.
- **Borda** loses most under strategy (86.0 / 94.7 → 85.1 / 91.7; Borda-ER 85.5 / 91.7), then Baldwin-Dowdall (86.0 / 98.1 → 84.2 / 95.8). Minimax, Schulze, Ranked pairs, Kemeny-Young and Baldwin also end up lower on both axes, but by at most 0.5 VSE and 1.4 WME.
- **Nomination** changes the honest result most under plurality-based methods (Plurality 35%, Supplementary vote 32%, TTR and contingent variants 30%, IRV 22%) and approval methods (21%), less under the rated and hybrid methods (8–13%, except MCA, MJ, 321, MDDA, ARA-Top and Reciprocal score at 17–23%) and least under the Condorcet methods (2.5–6%, except Baldwin-Dowdall 7.5% and Nanson-Dowdall 12%). Borda and Approval attract clones (0.65 and 0.81 per election).
- **Sincerity:** rated, approval and hybrid ballots stay at 99.4–99.9 (Reciprocal score 98.7); Condorcet methods at 96–98; Borda is the least sincere serious method (94.3).

## Equal ranking and two-round elections, poll rounds 1–6 (2026-10-09)

Run (user): `--rounds 1 2 3 4 5 6`, 400 elections, nomination on (`--exits 1 --clones 2`), defaults otherwise; the ER pairs plus IRV-3, TTR, Supplementary vote and the three approval methods. Paired differences per round.

- **Equal-ranking versions are mostly ties with their strict versions.** Honest values are identical (sincere ballots are strict). Under strategy, Minimax-ER, Schulze-ER, Baldwin-ER and Ranked Robin vs Ranked Robin-SR are ties in every round. ER-IRV is better than IRV twice (VSE\* +1.7 in round 1, WME\* +2.9 in round 4) and tied otherwise. Borda-ER vs Borda is a tie in every round, with the sign changing (−1.7 / −2.1 in round 1, +0.8 / +2.1 in round 3, −2.4 / −1.4 in round 6). The behaviour differs consistently: ER voters compromise less and bury more (Borda-ER 7–8% compromise and 24–28% burial vs 13–15% and 19% for Borda). Part of that burial is compromise by tying: a voter who ties the preferred leader with the favourite and moves the other leader down counts as burial, because the favourite stays on top. The prediction that ER versions are always better: equal or slightly better for IRV and the Condorcet methods, not shown for Borda.
- **Approval cond. runoff is equal to or better than Approval:** better on honest WME (+1.5) and, barely, on VSE\* in rounds 2 and 4 (+1.4 to +1.6); ties otherwise.
- **Always holding the runoff is better still:** Approval runoff beats Approval cond. runoff on WME\* in every round (+3.4 to +6.4), on honest WME (+4.2) and on CE\* (+8 to +12); VSE and VSE\* are ties. Ending the election at a majority of approvals gives away most of what the runoff adds, since that candidate often is not the head-to-head winner.
- **Borda is the only method that loses clearly from honest to round 1** (86.0 / 94.7 → 83.3 / 88.6; Borda-ER 81.6 / 86.5). The others stay put or gain; TTR gains WME\* now that burial is gone (88.5 → 89–92). Borda recovers partly with more rounds (87.2 / 93.9 in round 6).

## Model changes: separate runoffs, shared random streams (2026-10-09)

- **Two-round elections have a separate runoff.** TTR, Approval runoff and Approval cond. runoff used to read the runoff from the same ballot as the first round, so strategic voters could also misrank the finalists, which a real separate runoff gives no reason to do. Now the runoff is decided by sincere preferences and strategy changes only the round-1 marks; TTR also ends in round 1 when a candidate has a majority of the marks ([model.md](model.md), Two-round elections). **Every section below this one used the old versions:** what they call TTR is now **Contingent vote** (results unchanged), and the old Approval runoff and Approval cond. runoff no longer exist. The old TTR-ER, which is not an equal-ranking version of a two-round election, is now **Contingent-ER**.
- **Shared random streams.** Strategic nomination and the poll rounds used a random stream per method name; now every method uses the same one ([model.md](model.md), Random streams). Pairs that agree on honest ballots used to differ in nomination (e.g. TTR 85.9 / 88.9 vs TTR-ER 85.8 / 87.9, honest, with nomination); they now agree exactly. Results from before differ by sampling noise.

## Strategy model changed: poll rounds (2026-10-08)

**All strategic results above this line (and in the older sections below) used convergent learning:** over 20 rounds, groups of up to 10 voters with the same ranking took turns, each seeing the others' latest strategic ballots, and kept changes that paid. That describes where an electorate ends up after long adjustment, with far more information and coordination than one election offers. It protects methods whose strategy fails only when many use it at once.

The test that showed it (300 electorates, 200 voters, 6 candidates, defaults): when all voters bury at once, each putting the preferred of the two honest frontrunners first and the other last, Borda collapses from VSE 86.9 / WME 93.6 to 46.5 / 29.2 (the dark horse wins), while IRV, TTR, Minimax, Schulze, Baldwin, Ranked Robin and Black keep VSE about 85–86 and WME 97–99. Under convergent learning Borda kept VSE\* 87.6 / WME\* 94.4 (600 elections, full strategy), because burying groups saw the dark horse rise and backed off; allowing whole ranking groups instead of groups of 10 already lowered it to 81.1 / 86.2 (100 elections).

New default ([model.md](model.md), Strategy engine): round 0 is sincere; each poll round publishes only the method's counts from a sample of half the electorate, and all groups (whole ranking groups) answer at once; the strategic axes report round 1. Convergent learning stays available with `--converge 20 --samples 10`.

**First version, and what was wrong with it.** With polls only as uncertain as their sampling error, one round hit Approval and Score hard (40 elections, strategic share 0.5: WME 87 → 69 and 86 → 71). The runner-up's supporters dropped the leader, but the leader's supporters did not drop the runner-up. Two reasons, both about information, not strategy:

- In a 99-voter election the leader's lead (6 votes in the case examined) was far larger than the poll's sampling error (2.8), so the leader's supporters saw a poll in which they were behind in 0.4% of polls: nothing to protect against. The runner-up's ranking groups (8 and 12 voters) could overturn the lead outright.
- Most ranking groups have 1–3 voters. Their change flipped no poll, so their expected gain was exactly 0 and honesty won, although theory (Myerson & Weber 1993) has every voter act on the direction of a positive gain, however small.

**First fix (replaced):** wider poll uncertainty plus a minimum weight for small groups. It helped (leader rule on 68% / 77% of the two sides) but still depended on how close the race was: a leader's supporters protected the lead only when some poll showed it in danger.

**Second fix (replaced): "it helps when it first matters."** Each change was judged at levels with 4^k times wider polls and 4^k times as many voters making it, compared lexicographically. It made the leader rule independent of the gap, but it was slow (26 s per election and core), and changes among hopeless candidates, harmless in the polls as voters see them, got decided by wild scenarios at the outer levels: burial rose to 42–58% for almost every method in a full-strategy run (100 elections), Score included.

**Rule now: leaders and naive exaggerations.** The race that matters is found once per round for everyone: the two candidates that win most often over the polls; if one wins every poll, the polls are widened until another one wins somewhere. Each group then tries a few candidate ballots, the naive exaggerations of its ballot type around those two (approve down to the preferred leader; top score down to it, bottom from the other one, stretched in between; vote for it; move it to the top or the other one to the bottom of a ranking; …), plus its previous answer, and takes the best if it does not lose in the polls: exaggeration wins ties, since strategic voters exaggerate when it cannot hurt, whatever the gap. A few random changes follow, each needing a real gain ([model.md](model.md), Strategy engine). About 6.5 s per election and core with all methods.

Approval, same 40 elections, defaults (strategic share 0.5): among strategic voters who honestly approve both poll leaders, 98% of the leader's side and 76% of the runner-up's side approve only the one they prefer after one round, 88% for gaps under 10% of the voters and 87% over; approving only the preferred leader went from 25% of strategic voters (honest) to 78%; Approval VSE\* / WME\* 91.8 / 89.7 (honest 85.2 / 86.9).

Full strategy, one round, 40 elections (honest VSE / WME → VSE\* / WME\*, burial, compromise):

| Method | | burial | compromise |
|---|---|---|---|
| Plurality | 60.2 / 52.0 → 82.0 / 57.0 | 0% | 33% |
| IRV | 85.9 / 64.7 → 77.4 / 74.5 | 36% | 22% |
| Borda | 78.9 / 89.1 → 49.0 / 45.7 | 44% | 37% |
| Dowdall | 78.8 / 66.2 → 81.6 / 75.0 | 46% | 27% |
| Minimax, Schulze | 88.5 / 100 → 81–82 / 91 | 41% | 25% |
| Approval | 85.2 / 86.9 → 84.3 / 91.0 | 2% | 0% |
| Score | 93.6 / 85.9 → 88.6 / 94.8 | 3% | 0% |
| STAR | 94.2 / 92.7 → 90.9 / 88.2 | 12% | 1% |
| BTR-score | 92.7 / 95.2 → 88.0 / 94.0 | 10% | 0% |

The burial artefact is gone for rated ballots. Ranked ballots show about 40% burial by design: burying the other leader is the naive exaggeration and is taken whenever it cannot hurt in the polls. That sinks Borda (linear points: the dark horse wins) but not Dowdall (harmonic points: the bottom places hardly differ, and compromise, which helps, dominates); Condorcet methods lose some WME but keep 91. This is the naive first response; how it develops over more rounds is open.

## Aggressive share 0–1 × poll rounds 1–5, full strategy (2026-10-09)

Run (user): `--aggressive 0 0.25 0.5 0.75 1 --rounds 1 2 3 4 5 --strategic 1`, 200 elections per condition, defaults otherwise.

WME\* in round 1 by aggressive share (share of strategic voters who exaggerate whenever it cannot hurt; the others only when the polls show a gain):

| Method | 0 (all kind) | 0.25 | 0.5 | 0.75 | 1 (all aggressive) |
|---|---|---|---|---|---|
| MMWV | 91.1 | 92.2 | 94.3 | 89.9 | 79.5 |
| MMPOWV | 93.3 | 93.0 | 93.9 | 89.3 | 84.8 |
| Ranked Robin | 87.3 | 89.2 | 90.7 | 89.9 | 78.6 |
| Schulze | 96.3 | 96.2 | 95.8 | 94.8 | 90.0 |
| Baldwin | 95.9 | 96.7 | 95.7 | 95.3 | 93.9 |
| STAR | 84.9 | 90.9 | 92.9 | 93.6 | 87.4 |
| BTR-score | 90.9 | 96.1 | 96.7 | 95.8 | 90.8 |
| Score | 81.8 | 84.7 | 90.8 | 92.7 | 91.3 |
| Approval | 84.5 | 87.0 | 89.0 | 90.2 | 88.9 |
| IRV | 81.1 | 81.1 | 83.8 | 83.2 | 82.5 |
| Borda | 43.0 | 43.0 | 39.5 | 48.8 | 40.2 |
| Dowdall | 79.7 | 81.9 | 83.2 | 84.7 | 85.1 |

- **The extremes are the outliers.** For nearly every method the mixed settings (0.25–0.75) beat both ends, and between them most methods change little; by round 3 the spread narrows further (MMWV 94–96 for every mixed share).
- **A few kind voters are enough to stop mass burial.** The burial-prone Condorcet methods recover from all-aggressive to three quarters aggressive: MMWV 79.5 → 89.9, Ranked Robin 78.6 → 89.9, MMPOWV 84.8 → 89.3. Blind burial needs nearly everyone to do it.
- **A few aggressive voters help the rated methods.** All-kind is the worst setting for Score (81.8), STAR (84.9), BTR-score (90.9) and Approval (84.5); a quarter aggressive voters lifts them (BTR-score 96.1, STAR 90.9). Aggressive voters protect a lead whatever the gap; with only kind voters the runner-up's side attacks when the polls show a chance and the leader's side does nothing (see the leader rule above).
- **Borda collapses at every share**, all-kind included: its burial pays in the polls, it is not only naive.
- Reading: each extreme is a stress test for one weakness, blind mass burial (all aggressive) or unprotected leads (all kind). Real electorates are mixed, and the conclusions do not hinge on the exact share in between; 0.5 stays the default.

## Poll rounds 1–5, full strategy; aggressive and kind voters (2026-10-09)

Run (user): `--rounds 1 2 3 4 5 --strategic 1`, 200 elections, defaults otherwise (all strategic voters aggressive, the only behaviour at the time).

- **Burial under Condorcet methods is large**, something convergent learning missed: 55–65% of voters for MMPO, MMWV, MMPOWV, Ranked Robin, SICT; about 45% for Schulze, Baldwin, Black. Burying the other leader is the naive exaggeration, taken whenever it cannot hurt in the polls.
- **The methods that held up best before suffer most in round 1:** WME\* 79.5 (MMWV), 84.8 (MMPOWV), 78.6 (Ranked Robin), against 90–94 for Baldwin, Schulze and MMPO; STAR 87.4. With more rounds they recover: by round 3, WME\* 94.3 (MMWV), 93.6 (MMPOWV), 94.3 (Ranked Robin); STAR 93.6.
- **More rounds help most methods**; the order of methods on WME\* changes a lot between rounds (rank correlation with round 1: 0.07–0.26, noise reference 0.75).
- **Borda** collapses in round 1 (WME\* 40) and partly recovers (61–68); **Dowdall** gains from strategy (VSE 83 → 88–90, WME 75 → 84–87). Since Borda's weakness under naive strategy comes from its linear points, the Borda-based methods (Black, Baldwin, Nanson, BTR-Borda; Ranked Robin's tiebreak) might do better with Dowdall points; to try once the strategy model is settled.

**Aggressive vs kind voters.** Whether real voters bury when it buys nothing is doubtful, so `--aggressive` sets the share of strategic voters who exaggerate whenever it cannot hurt (aggressive); the others (kind) change their ballot only when the polls show a gain. Default 0.5. A first look (40 elections, full strategy, rounds 1 and 3; efficiencies noisy, burial shares clear):

| Burial | all aggressive | half | all kind |
|---|---|---|---|
| MMWV, MMPOWV | 58–63% | 33–37% | 17–23% |
| Schulze, Baldwin | 39–46% | 22–28% | 13–18% |
| IRV | 38–39% | 20–22% | 4% |
| STAR | 13–16% | 8% | 10–14% |
| Approval | 2–4% | 3% | 2–3% |

What remains with only kind voters is burial that pays in the polls. Kind voters also lose the leader rule's protective half (a leader's supporters see no gain in dropping the runner-up unless a poll shows danger), so under Approval the share mainly decides how strongly leads are protected.

## Defaults changed: width 1.0, 5 clusters (2026-10-08)

Earlier sections used width 1.5 and 3 clusters; from now on the defaults are width 1.0 and 5 clusters (spread stays 0.5). Based on two one-at-a-time sweeps, 400 elections each, seed 1, with exits 1 and clones 2:

- **Width** (0.25–2.0, 5 clusters, 6 candidates). Width only changes how utility falls with distance, not who prefers whom: the Condorcet winner share (96.8%) and the honest WME of ranked methods do not move (MMPO 99.6, IRV 86.0 at both 1.0 and 1.5). It acts on VSE, on rated and approval ballots, and on what strategy is worth. A wide width makes utility nearly linear in distance, so total utility and the majority mostly agree and the VSE axis is compressed: electing the Condorcet winner loses 12.9% of VSE at 1.0 and 8.5% at 1.5, the best-WME end of the frontier is at VSE 86.5 and 91.0, and the honest VSE gap between Score and MMPO is 6.7 and 4.3 points. Favourite ÷ least liked (median voter) is 4.4 at 1.0 and 2.7 at 1.5; very small widths are extreme (378 at 0.25, where electing the Condorcet winner loses 47% of VSE).
- **Clusters** (1, 2, 3, 4, 5, 10; 8 candidates, width 1.5). One cluster leaves little to separate the methods (electing the Condorcet winner loses 1.6% of VSE, frontier best-WME end at VSE 98.3); the tension grows up to 5 clusters (8.2%, VSE 91.4) and eases again at 10 (7.1%, VSE 92.7), where many small clusters blur into a more even electorate. No even/odd pattern beyond noise: the more stable electorate measures move steadily from 3 to 5 clusters (VSE lost 6.6, 7.1, 8.2%; frontier VSE 93.5, 92.3, 91.4), and only the share of Condorcet winners with the highest total utility zigzags (72.7, 75.5, 67.7%), by about the size of its sampling error.

## Score range 0–5, 0–10, 0–100 (2026-10-07)

Settings: the defaults (99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5, 400 elections per condition, seed 1; `--max-score 5 10 100`. Methods: Score, MJ, STAR, BTR-score, MARS, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score, and as fixed references 321 and MCA (three grades, unchanged across conditions, as they should be). The electorates are the same in all three conditions.

Honest VSE / WME → learned VSE\* / WME\*:

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
- **Score:** honest values do not change at all (rounding to 0–5 already averages out over voters). The learned VSE\* rises (91.9 → 92.8 → 93.8), WME\* stays about level. Not yet checked with a paired comparison across conditions; and since the score mutation differs above 0–5 ([model.md](model.md), Strategy engine), part of the change may come from the learning rather than the range.
- **Hybrids move along the frontier rather than toward it:** a wider range trades VSE for WME (STAR 95.4 / 95.6 → 94.6 / 99.0; MARS 95.8 / 95.6 → 95.0 / 99.3). With honest ballots that brings STAR and MARS close to the frontier at the WME end: the straight line from the best-WME end to the balanced point, which the frontier cannot lie below, gives VSE 94.5 at WME 99.3, and MARS has 95.0. Under strategy they stay well inside it (MARS* 93.9 / 98.1).
- **Score+ gains a little** (WME 92.6 → 94.2), consistent with its re-stretching losing less information on a finer range.
- **Methods that suffer from a finer range:** Reciprocal score (honest WME 85.5 → 80.2), MJ (WME\* 88.6 → 84.0) and ARA-Top (VSE 91.4 → 89.1). For ARA-Top this is by construction: on 0–100 practically only the favourite gets the top score, so it approaches a plurality-based approval count. Reciprocal score works with factions of top-rated candidates, which shrink to single favourites as well.

## Number of candidates, 3 to 9 (2026-10-07)

Settings: the defaults (99 voters, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5, 400 elections per condition, seed 1; base 3 candidates, swept over 3–9 (`--cands`).

**Electorates.** More candidates make the task harder for every method: a Condorcet winner exists in 99.8% of elections with 3 candidates and 94.8% with 9; it also has the highest total utility in 88.5% and 68.6% of those; preferences get steeper (favourite ÷ least liked, median voter, 1.7 → 3.0).

**Frontier.** The balanced point stays at about 97 VSE / 98 WME throughout. The best-WME end moves between VSE 91 and 94.5 without a trend; the best-VSE end loses some WME (89.7 → 87.5).

Honest VSE / WME → learned VSE\* / WME\*:

| Method | 3 candidates | 6 candidates | 9 candidates |
|---|---|---|---|
| Plurality | 75 / 83 → 88 / 91 | 52 / 55 → 82 / 75 | 51 / 46 → 78 / 66 |
| TTR | 93 / 92 → 94 / 98 | 87 / 75 → 90 / 90 | 86 / 70 → 90 / 84 |
| IRV | 93 / 92 → 94 / 99 | 90 / 78 → 92 / 92 | 91 / 72 → 91 / 85 |
| ER-IRV | 93 / 92 → 93 / 98 | 90 / 78 → 91 / 90 | 91 / 72 → 93 / 84 |
| STAR | 95 / 98 → 94 / 99 | 95 / 96 → 95 / 96 | 97 / 91 → 96 / 92 |
| MARS | 95 / 99 → 94 / 98 | | 96 / 92 → 96 / 93 |
| Smith//Score | 95 / 98 → 95 / 99 | | 96 / 92 → 95 / 95 |
| Minimax, Schulze, MMPO | VSE 91–94, WME 100, WME\* 99–100 at every count | | |

(Empty cells: not taken from the report.)

- **Plurality-based methods lose most** as candidates are added (vote splitting): honest WME drops from 92 to 70–72 for TTR and IRV, and to 46 for Plurality. Learned ballots recover part of it (voters compromise), but not all.
- **Condorcet methods are very stable:** WME 100 with honest ballots at every count and 99–100 learned; VSE stays in the 91–94 band of the best-WME end of the frontier.
- **Hybrids trade WME for VSE:** STAR, MARS and Smith//Score gain about 2 VSE and lose 6–7 WME from 3 to 9 candidates, consistent with the rated-ballot cap (more candidates share a rating on each ballot, hiding more preferences).
- **ER-IRV against IRV:** ER-IRV is more sincere at every count (e.g. 98.1 against 96.4 with 6 candidates, 98.5 against 96.6 with 9). The strategic axes are mixed: VSE\* higher with 9 candidates (93.0 against 91.3), lower with 6 (91.4 against 92.1), and WME\* often lower at high counts (85.9 against 89.8 with 8). Equal rankings let voters support a compromise without dropping their favourite, which shows in sincerity; it is not a clear gain in outcomes.
- **Rank stability** against 3 candidates: sincerity stays at the noise reference (0.80–0.92 against 0.81). WME and WME\* stay high (0.67–0.94) but mostly below their references (0.95, 0.93), so the order shifts a little, in line with the plurality-based methods falling back. With 3 candidates the methods are so close on VSE and VSE\* that the noise reference is low (0.29 and −0.01), so the order there is mostly noise.

## MMPO family under full strategy (2026-10-07)

Settings: 200 voters, 6 candidates, otherwise the defaults; strategic share 1.0, 800 elections, seed 1. Methods MMPO, MMWV, Minimax product, MMPOWV.

- All four are identical with honest ballots (VSE 92.3, WME 100, CE 100).
- Learned: VSE\* 91.9–92.2, WME\* 99.1 (MMPOWV) to 99.5 (Minimax product), CE\* 98.2–98.4, sincerity 99.8 for all; burial 1.7–1.9% of voters, compromise 0.5–0.6%, push-over 0.
- **Every paired difference is a tie** (MMPOWV's WME\* is 0.2–0.4 below the others, within noise). The slight tendency for MMPOWV seen in an earlier run did not hold up. MMPOWV replaces Minimax product because it is the simpler form (two factors instead of three) with no measurable loss, not because it is better.

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

- **Minimax product** (margins × winning votes × pairwise opposition, each as a share 0–1): equal to Minimax with honest ballots, a tie at strategic share 0.5, and at full strategy clearly better than Minimax on WME\* (+2.1), CE\* (+4.5) and sincerity (+1.7). MMPO, MMWV and Minimax product are within each other's noise (WME\* at full strategy 99.0, 98.7, 99.1). The variant without the margin factor, **MMPOWV** (pairwise opposition × winning votes), replaces Minimax product; see the next section.
- **Minimax+Approval** (approvals + worst margin) and **Approval product** (winning votes × opposition × approvals, on ARA ballots): undominated, but inside the hybrid cluster. Minimax+Approval ties STAR on everything except WME\* at full strategy (+1.8); Approval product trades 0.7 VSE for 0.8 WME against STAR. Neither moves toward the balanced point of the frontier; both were removed.
- **Under full strategy** the Condorcet methods that measure opposition (MMPO, MMWV, Minimax product) are the most honest of all (sincerity 99.6–99.7 against 98.0–98.5 for the hybrids, burial about 3% of voters against 15–22%).

## Margin band, a negative result (2026-10-07)

Margin band (proposed in this lab, removed again): every candidate whose worst margin is within 5% of the voters of the best worst margin is in a band; the band member with the highest score total wins. Same settings as above, 1200 elections per condition, compared with Score, STAR, BTR-score, Smith//Score, MARS, Minimax, Schulze, Ranked pairs and IRV.

- It tied Smith//Score on every axis at both strategy levels: with a Condorcet winner in 98% of elections, the band and the Smith set nearly always pick the same candidate.
- The hoped-for burial resistance did not appear: 17.5% of voters buried at full strategy, about as many as with the other hybrids, against about 6% with Schulze, Minimax and Ranked pairs.
- **Under full strategy the margin-based Condorcet methods weakly dominate the hybrids:** the hybrids' VSE advantage holds only with honest or partly strategic voters. At full strategy all hybrids drop to VSE\* 92.2–92.8, level with Schulze, Ranked pairs and Minimax (92.5–93.1), while the Condorcet methods keep the clearly higher WME\* (97–98 against 93–95).
- **Rated ballots cap WME** (on 0–5; with 0–100 the cap disappears, see "Score range"): every method taking pairwise preferences from 0–5 ratings (Smith//Score, BTR-score, STAR, Margin band) reached only about 95 WME with honest ballots. Equal ratings hide preferences, so even a Condorcet rule on ratings misses some sincere Condorcet winners.

## Current picture (6 candidates, Zipf, 2026-10-06)

```
python sweep.py --one-at-a-time --zipf 1 --dims 10 --clusters 3 5 --spread 0.5 0.75 \
    --width 1.5 --cands 6 --group-size 10 1 0 --seeds 1 2 --elections 400 --out model \
    --strategic 0.5 1 --rounds 20 --exclude "Condorcet-random-drop" "Reciprocal score"
python analyze.py model-elections.csv --out model-analysis
```

Base: 99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, strategic share 0.5, 20 rounds; 800 elections per condition. These are now the defaults.

Methods that no other method clearly beats on PE, WM, PE\* and WM\* (out of 6 conditions):

| Method | Undominated in |
|---|---|
| Schulze, MARS | 6 |
| Minimax, Ranked pairs, BTR-score | 5 |
| STAR | 4 |
| all others | never |

- **Two groups lead:** margin-based Condorcet methods (Schulze, Ranked pairs, Minimax) and score runoffs (MARS, BTR-score, STAR). Score runoffs are ahead on PE, the Condorcet methods on WM.
- **Burial sensitivity separates the Condorcet methods.** Their honest values are identical, but Benham, BTR-IRV, SICT and Copeland lose 1–2 points more WM under full strategy than Schulze, Ranked pairs and Minimax.
- **Plurality-based methods gain from strategy, but never enough.** Under full strategy TTR's WM rises from 6.5 (honest) to 13.7 and IRV's from 7.3 to 13.2, while Score drops from 13.0 to 10.9; the margin-based Condorcet methods lose least (15.7 → 14.5–14.9).
- **What changes the method order:** full strategy (PE\* rank agreement 0.46 against 0.79 from noise), 5 clusters (0.56) and group size 1 (0.58). Spread 0.75 does not.
- **Cycles stay rare** with 6 candidates too (1.6–3% of elections), so how a method resolves cycles matters less here than how well it resists burial.
- **Group size changes little with 6 candidates,** partly for a mechanical reason: groups are voters with the same full ranking, and with 720 possible rankings most groups are small whatever the cap (see [model.md](model.md), Coordination).

The same kind of sweep with 4 candidates (width 1.0 1.5 2.0, spread 0.5 0.75 1.0, clusters 3 2 5, strategic share 0.5, 800 elections per condition) gave a similar but wider set: Schulze (undominated in 6 of 7 conditions), STAR, MARS, BTR-Borda (5), Minimax (4), BTR-score, SICT, Baldwin, Approval runoff (2), ARA, Score (1). Width hardly changed the order there; spread and 2 clusters did, mainly under strategy.

## Voter model (2026-10-05/06)

**Plain dimensions, one parameter at a time** (99 voters, 4 candidates, group size 10, strategic share 0.5, 15 rounds, 200 elections per condition): most apparent reordering was within sampling noise, especially on PE\*. The electorates themselves (1500–2000 generated per setting) explained most of the rest:

| Setting | Condorcet winner also highest total utility | PE lost electing it | Favourite ÷ least liked (median) |
|---|---|---|---|
| base (2 dims, width 1) | 80% | 7.7% | 3.9 |
| dims 4 | 76% | 10.9% | 6.0 |
| width 0.5 | 74% | — | 17 |
| width 2.0 | 85% | — | 2.0 |

(The PE-lost figure is a ratio of sums, as in the report.) More dimensions and narrower width make the Condorcet winner less often the best candidate by total utility, which lowers PE for Condorcet and plurality-based methods and favours rated methods. Width 0.5 makes preferences implausibly steep (a voter values the favourite 17 times the least liked candidate).

**Zipf dimensions** (dimension i with standard deviation 1/i, after Chris Smith) make the number of dimensions almost irrelevant: from about 4 dimensions on the electorates are the same within noise.

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

| Pair | PE | CE | PE\* | CE\* | dissonance |
|---|---|---|---|---|---|
| ACR − Approval | tie | tie | tie | tie | +2.8 |
| Approval runoff − Approval | +2.0 | +19.2 | tie | +9.4 | +1.4 |
| Approval runoff − TTR | +3.3 | +20.4 | +1.5 | tie | −1.3 |
| ACR − Approval runoff | −2.2 | −19.2 | −1.6 | −9.0 | +1.4 |
| ACR − TTR | tie | tie | tie | −7.5 | tie |

("Dissonance" was the complement of today's sincerity.)

- With honest ballots the most approved candidate had more than half the approvals in 97% of elections, and the top two both did in 84%. So the conditional runoff almost never happens: ACR is Approval, plus a threshold that invites strategy.
- A majority of first choices implies a majority against every rival; a majority of approvals does not. The 50% rule checks for a real majority in plurality, but almost nothing in approval.
- Approval with an obligatory runoff is better than TTR on three of four axes, ties on CE\*, and has less dissonance. If a 50%-winner is required, that is the version to use.
- Conditional TTR needs no separate test: in the model it is identical to TTR ([model.md](model.md), Strategy engine).

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

- **Rounds:** with 50 voters and 6 candidates, PE\*, CE\* and sincerity changed only within noise between 20 and 40 rounds; at 10 rounds they were still moving. 20 rounds is the default.
- **No early stop:** stopping once a round passes without any accepted change was tested (200 elections, 99 voters, 4 candidates, plain 2 dimensions, whole ranking groups, full strategy). It saves most of the run time, but quiet rounds are common long before learning has finished. Stopping after one quiet round changed 55 of 99 method × axis values clearly (sincerity higher for almost every method, CE\* up to 9 points higher for Condorcet methods); after two quiet rounds still 50.
- **Honesty wins ties:** without it, deviations that stopped paying off stayed, and dissonance grew with the number of rounds.
