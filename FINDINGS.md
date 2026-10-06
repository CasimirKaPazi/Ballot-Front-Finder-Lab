# Voting Pareto Lab — findings

What the simulations have shown so far. The model, parameters and definitions are in [README.md](README.md). Newest first; each section names the settings it used, since the model and defaults changed along the way (plain dimensions before Zipf, CE as an axis before WM). Numbers are point estimates; differences only count where the paired comparison or the noise reference says so.

**Units.** Up to 2026-10-06 the axes were PE (zero = utility-proportional lottery) and WM (the winner's worst margin in % of voters); the sections below keep those units. Since then they are VSE and WME, both with 0 = random ballot and 100 = best candidate (README, Axes). Within one condition the change is a fixed rescaling of each axis (the sums over elections that set the zero and the 100 are the same for all methods), so method order, dominance and rank stability are unaffected; only the numbers differ.

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
- **Plurality-based methods gain from strategy, but never enough.** Under full strategy Top-two runoff's WM rises from 6.5 (honest) to 13.7 and IRV's from 7.3 to 13.2, while Score drops from 13.0 to 10.9; the margin-based Condorcet methods lose least (15.7 → 14.5–14.9).
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

In Germany a winner is expected to have more than half the votes. The tempting chain "use approval instead of top-two runoff, and add a runoff only when nobody has a majority" does not work in the model.

Settings: 600 elections, 200 voters, 6 candidates, plain 2 dimensions, 3 clusters, spread 0.5, width 1.0, group size 10, full strategy, 20 rounds (CE was still an axis then). Paired differences, percentage points (ties in brackets are within noise):

| Pair | PE | CE | PE* | CE* | dissonance |
|---|---|---|---|---|---|
| ACR − Approval | tie | tie | tie | tie | +2.8 |
| Approval runoff − Approval | +2.0 | +19.2 | tie | +9.4 | +1.4 |
| Approval runoff − Top-two runoff | +3.3 | +20.4 | +1.5 | tie | −1.3 |
| ACR − Approval runoff | −2.2 | −19.2 | −1.6 | −9.0 | +1.4 |
| ACR − Top-two runoff | tie | tie | tie | −7.5 | tie |

("Dissonance" was the complement of today's sincerity.)

- With honest ballots the most approved candidate had more than half the approvals in 97% of elections, and the top two both did in 84%. So the conditional runoff almost never happens: ACR is Approval, plus a threshold that invites strategy.
- A majority of first choices implies a majority against every rival; a majority of approvals does not. The 50% rule checks for a real majority in plurality, but almost nothing in approval.
- Approval with an obligatory runoff is better than Top-two runoff on three of four axes, ties on CE*, and has less dissonance. If a 50%-winner is required, that is the version to use.
- Conditional top-two runoff needs no separate test: in the model it is identical to top-two runoff (README, Strategy engine).

## Coordination: burial against Condorcet methods (2026-10-03)

Plain 2 dimensions, 99 voters, 4 candidates, full strategy, 60 elections; CE with learned ballots:

| Method | Whole ranking groups | Every voter alone | Burial (groups / alone) |
|---|---|---|---|
| Top-two runoff | 94.9 | 94.1 | 0.1% / 0.3% |
| IRV | 96.6 | 91.5 | 1.8% / 1.0% |
| Copeland | 83.6 | 100.0 | 20.1% / 2.0% |
| Minimax | 89.8 | 100.0 | 20.6% / 1.3% |
| BTR-Borda | 91.5 | 100.0 | 23.2% / 1.2% |
| BTR-score | 89.8 | 96.6 | 20.5% / 2.8% |
| STAR | 89.8 | 98.3 | 18.6% / 3.1% |

Burial needs blocs; compromise works for individuals. With organised factions, Condorcet methods and score runoffs lose much of their Condorcet efficiency to burial, while Top-two runoff and IRV are carried toward the Condorcet winner by compromise. With individual voters the picture reverses. This is why every strategic result must be read with its group size.

## Strategy engine checks (2026-10-03/04)

- **Rounds:** with 50 voters and 6 candidates, PE*, CE* and sincerity changed only within noise between 20 and 40 rounds; at 10 rounds they were still moving. 20 rounds is the default.
- **No early stop:** stopping once a round passes without any accepted change was tested (200 elections, 99 voters, 4 candidates, plain 2 dimensions, whole ranking groups, full strategy). It saves most of the run time, but quiet rounds are common long before learning has finished. Stopping after one quiet round changed 55 of 99 method × axis values clearly (sincerity higher for almost every method, CE* up to 9 points higher for Condorcet methods); after two quiet rounds still 50.
- **Honesty wins ties:** without it, deviations that stopped paying off stayed, and dissonance grew with the number of rounds.
