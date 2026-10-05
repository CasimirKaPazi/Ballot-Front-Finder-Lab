# Voting Pareto Lab

Python 3 + NumPy; the analysis also needs matplotlib. Honest and learned-strategy elections, evaluated on a Pareto front.

```
python test_methods.py                                   # hand-built checks
python run.py --elections 200                            # one condition, prints a table
python sweep.py --group-size 1 10 0 --seeds 1 2          # grid of conditions -> sweep-elections.csv
python analyze.py sweep-elections.csv --compare "BTR-score:STAR"   # -> analysis/
```

300 elections with all 31 methods take about 8 minutes on 2 cores (default condition; 200 voters and 6 candidates take about 5 times as long); time scales linearly with cores. Use `--methods` or `--exclude` to run fewer, by name or by group (see Methods).

## Files

| File | Contents |
|---|---|
| `scenario.py` | Electorate generator: spatial clusters, absolute utilities `exp(-distance / width)` |
| `methods.py` | Ballot types, sincere ballots, exact tallies on ballot matrices (ties shared) |
| `strategy.py` | Learning engine, sincerity and strategy types |
| `metrics.py` | Efficiencies for three baselines, winner's worst margin, Condorcet efficiency, Pareto front |
| `run.py` | Tallies every electorate with sincere and learned ballots, runs in parallel, writes CSV with the run's parameters as columns |
| `sweep.py` | Runs a grid of conditions and seeds into one CSV |
| `analyze.py` | Paired bootstrap, intervals, dominance pointers, paired comparisons, Markdown report |
| `plots.py` | The analysis plots |
| `test_methods.py` | Hand-built checks |

## Strategy engine

- **Groups:** voters with the same sincere ranking, optionally split at random into subgroups of at most `--group-size` voters (see Coordination below). A group either votes sincerely (each member their own sincere ballot) or casts one shared ballot.
- **Polls:** each round draws 10 bootstrap resamples of the electorate. A proposal is judged by the group's total expected utility averaged over these samples, so a group gains in proportion to how often it would be pivotal, not only when it is decisive in the exact electorate.
- **Proposals:** one small change to the group's ballot:
  - one name: switch to another candidate
  - ranking: swap two candidates
  - approval: toggle one candidate (never empty or full)
  - ranking with approval line: swap two candidates, or move the line up or down by one
  - veto: move the veto to another candidate
  - ranking with ties: move one candidate to any other level (equal levels are ties)
  - truncated ranking: swap two candidates, or rank one more or one fewer
  - score: set one candidate to a different value 0–5

  Voters start sincere. A group that has not deviated yet changes the sincere ballot of its average member (mean utilities of the group); a deviating group changes its current shared ballot, and sometimes (15% of proposals) proposes going back to sincere.
- **Acceptance:** a proposal is kept only if it raises the group's expected utility. A return to sincere is also kept when it is not worse: voters prefer honesty when indifferent. Without this, a deviation that once paid off stays forever once it stops mattering, and sincerity keeps falling with the number of rounds instead of reflecting what strategy actually pays.
- **Rounds:** 20, groups in random order each round. In a test with 50 voters and 6 candidates, PE*, CE* and sincerity changed only within noise between 20 and 40 rounds; at 10 rounds they were still moving. Stopping early once a round passes without any accepted change was tested and rejected (200 elections, default condition): it saves most of the run time, but quiet rounds are common long before learning has finished, since each group tries only one random change per round. Stopping after one quiet round changed 55 of 99 method × axis values clearly (sincerity higher for almost every method, CE* up to 9 points higher for Condorcet methods); after two quiet rounds still 50. For the same reason, `settled` (no change in the last round) is a weak sign of convergence.
- **Feature sums:** every tally except Condorcet-random-drop depends on the ballots only through sums over voters of per-voter features (totals, first and last places, the pairwise matrix, grade counts, first choices among each subset of candidates, ...). Each method lists its features and decides from their sums. During learning the sums are kept per poll sample and updated only for the group that changes. All features are integers, so this gives exactly the same learned ballots as recounting (checked against the earlier code).
- **Conditional top-two runoff** is not a separate method: a candidate with more than half the first choices also wins every runoff, since the runoff reuses the same rankings. (In a real two-round election voters vote again, with new information and different turnout; that is not modelled.)
- **Random streams:** each method learns with its own random stream per election, keyed by the method's name, so results do not change when methods are added or reordered.

## Electorates

Each election draws a fresh electorate (`scenario.py`):

- cluster centres: `--clusters` points, standard normal in `--dims` dimensions;
- cluster sizes: random weights from a flat Dirichlet distribution, so some elections have one dominant cluster and others are balanced;
- voters: each assigned to a cluster by these weights, placed normally around its centre with standard deviation `--spread`;
- candidates: drawn from the same distribution as voters (cluster by the same weights, same spread);
- utilities: `u = exp(-distance / width)`, absolute and not normalised: 1 at zero distance, towards 0 far away. A smaller `--width` makes voters care only about nearby candidates.

## Parameters

`run.py` (one condition):

| Flag | Default | Meaning |
|---|---|---|
| `--elections` | 200 | electorates |
| `--voters` | 99 | voters per election (odd, so pairwise majorities rarely tie) |
| `--cands` | 4 | candidates |
| `--dims` | 2 | issue-space dimensions |
| `--clusters` | 3 | voter clusters |
| `--spread` | 0.5 | standard deviation of voters around their cluster centre (centres have 1) |
| `--width` | 1.0 | distance scale of the utility fall-off |
| `--seed` | 1 | random seed; election e uses the stream (seed, e), so all methods face the same electorates |
| `--rounds` | 20 | learning rounds |
| `--samples` | 10 | poll resamples per round |
| `--group-size` | 0 | most voters that coordinate as one group; 0 = whole ranking group, 1 = every voter alone |
| `--strategic` | 1.0 | share of voters who vote strategically (see Partial strategy) |
| `--workers` | all cores | parallel processes |
| `--methods` | all | run only these methods (start from none); method or group names (see Methods), case ignored |
| `--exclude` | none | run all methods except these |
| `--out` | `results` | output prefix: `<out>-elections.csv` and `<out>-summary.csv` |

`sweep.py` takes the same flags, but every condition flag (`--voters` … `--width`, `--rounds`, `--samples`, `--group-size`, `--strategic`) accepts several values, and `--seeds` several seeds; it runs the full grid and writes `<out>-elections.csv` (default `--out sweep`), one run after another.

`analyze.py`:

| Flag | Default | Meaning |
|---|---|---|
| (positional) | | one or more elections CSVs |
| `--out` | `analysis` | output directory |
| `--boot` | 1000 | bootstrap replicates |
| `--compare` | none | pairs `"A:B"` for paired differences |
| `--hide` | Random ballot, Random pair runoff, Anti-plurality, TRASH | methods left out of the scatter plots; `--hide` alone shows all |
| `--seed` | 0 | bootstrap random seed |

## Coordination: what the strategic axes measure

How many voters move together decides which strategies pay, and the methods differ sharply in this.

- **Compromise** (voting for a viable candidate instead of a hopeless favourite) pays even for a single voter. It is the main strategy in Top-two runoff and IRV, and it tends to carry the Condorcet winner into the final round.
- **Burial** (ranking or rating a rival below a candidate one likes less, to create a cycle or to drop the rival in a runoff) only pays when many voters do it together. A single voter out of 99 is almost never pivotal for it. It is the main strategy against Condorcet methods, STAR and BTR-score, and it targets exactly the Condorcet winner.

Diagnostic run (before WM was introduced), 60 elections, default scenario, CE with learned ballots:

| Method | whole ranking groups | every voter alone | burial (groups / alone) |
|---|---|---|---|
| Top-two runoff | 94.9 | 94.1 | 0.1% / 0.3% |
| IRV | 96.6 | 91.5 | 1.8% / 1.0% |
| Copeland | 83.6 | 100.0 | 20.1% / 2.0% |
| Minimax | 89.8 | 100.0 | 20.6% / 1.3% |
| BTR-Borda | 91.5 | 100.0 | 23.2% / 1.2% |
| BTR-score | 89.8 | 96.6 | 20.5% / 2.8% |
| STAR | 89.8 | 98.3 | 18.6% / 3.1% |

So with the default (`--group-size 0`, whole ranking groups) the strategic axes measure resistance to **organised factional strategy**: every voter with the same ranking follows one recommendation. That is the case most favourable to burial. With `--group-size 1` they measure resistance to **individual strategy**, where burial almost never pays. Reality lies in between, and results under strategy should always be read together with the group size used.

A second, smaller asymmetry: learning is myopic. A burying group does not anticipate that the Condorcet winner's supporters may bury back, or that the cycle it creates can elect a candidate it likes even less (the chicken dilemma). This probably makes burial look somewhat more attractive than it would be to far-sighted voters.

Smaller groups mean more groups, and run time grows with their number: `--group-size 1` takes roughly six times as long as the default.

## Partial strategy

`--strategic p` makes a random share p of the voters strategic; the others always vote sincerely. Each voter is strategic with probability p, drawn from its own random stream per election, so the same voters are strategic for every method. Groups (and their size cap) form only among the strategic voters. p = 1 (the default) gives exactly the results of full strategy; p = 0 gives the sincere results. Sincerity and the strategy types are averaged over all voters, including the sincere ones.

This follows the idea of intermediate strategy levels in other simulations without further assumptions about who becomes strategic. Run time falls less than in proportion: halving the strategic voters removes fewer than half of the groups, since many groups are small (30 elections, default condition: p = 0.5 took about 80% of the time of p = 1).

## Axes and other outputs

**Efficiency** against a baseline lottery q, as a ratio of sums over all elections:
`E = Σ(U_method − U_base) / Σ(U_best − U_base)`, where U is a candidate's total utility, U_method the expected total utility of the method's winner and U_best the best candidate's. The baseline: each voter's personal lottery proportional to `u^τ`, averaged over voters.

| Axis | τ | Baseline |
|---|---|---|
| VSE | 0 | random candidate |
| PE | 1 | utility-proportional lottery (headline) |
| RBE | ∞ | random ballot |

Within one condition the three rank methods identically (same numerator, different zero point), so only PE is used as an axis. No utility normalisation: a voter who cares little (all candidates far away) counts little.

**Winner's worst margin (WM)**, the second axis, in % of voters: the elected candidate's smallest head-to-head margin against any other candidate, by sincere preferences, averaged over elections (weighted by the winner probabilities). In short: how clearly the winner would survive the hardest two-way race.

- Positive: the winner is the Condorcet winner, by that much. Negative: a majority prefers some specific rival, by that much.
- It is a refined Condorcet measure. The motivation is legitimacy: after the election voters judge the result, and if someone other than the Condorcet winner won, they can argue that a majority would have preferred another candidate. WM measures how strong that argument is, instead of only whether it exists: losing a Condorcet winner by one vote costs little, electing a candidate whom a large majority prefers a rival to costs a lot.
- Every election counts, cycles included, and the value does not depend on the number of voters.
- With honest ballots Minimax maximises WM by construction (it elects the candidate with the best worst margin), just as all Condorcet methods reach 100 on CE. The strategic version WM* is where methods are tested.

**Condorcet efficiency (CE)**, reported alongside but no longer an axis: among elections with a Condorcet winner by sincere utilities, the probability the method elects it. It ignores elections without a Condorcet winner and does not distinguish narrow from clear ones.

PE, WM and CE use sincere ballots; PE*, WM* and CE* (in the CSV: PE_learned, WM_learned, CE_learned) the learned ballots. In all cases the preferences that count are the sincere ones.

**Output files of `run.py`:** `<out>-elections.csv` has one row per (election, method, ballots = sincere or learned) with the run's parameters as columns, the utilities needed for the axes (U_method, U_best, U_tau0, U_tau1, U_tauinf), has_cw, p_cw (probability of electing the Condorcet winner), WM (winner's worst margin, share of voters), sincerity, settled and the strategy types. `<out>-summary.csv` and the printed table have one row per method:

| Column | Meaning |
|---|---|
| PE | proportional efficiency, sincere ballots |
| WM | winner's worst margin, sincere ballots |
| PE_learned, WM_learned | the same with learned ballots |
| CE, CE_learned | Condorcet efficiency, sincere and learned ballots |
| VSE, RBE | efficiency against random candidate and random ballot, sincere ballots |
| sincerity | mean sincerity of the learned ballots, over all voters (below) |
| settled | share of elections where no group changed its ballot in the last round; cycles never settle |

`on_front` in the summary is a quick check on point estimates (PE, WM, PE_learned, WM_learned, no uncertainty). The analysis uses the tie-aware "Dominated by" instead (see Analysis).

**Strategy types** (`strategy.strategy_types`), among voters whose learned ballot reverses at least one real preference:
- *compromise*: the favourite is not at the top, and the best-liked top candidate is liked above the voter's mean (a viable second choice raised);
- *push-over*: the favourite is not at the top, and every top candidate is liked below the mean (a weak candidate raised, to steer eliminations or runoffs);
- *burial*: the favourite stays at the top, but a preference further down is reversed.

The three are exclusive. For score ballots, "burial" includes small reversals among lower-rated candidates, so a high burial share can go with high sincerity.

**Sincerity** (`strategy.sincerity`), per voter: for every pair the ballot orders (a above b) while the voter likes b more, add `u[b] − u[a]`; divide by the sum of `|u[a] − u[b]|` over all pairs; sincerity is 1 minus this share. 1 (100%) means the ballot never claims a preference the voter does not hold (true for every sincere ballot, since rounding and leaving candidates level never reverse an order). 0 means every preference is reversed. Example, utilities 0.5 / 0.3 / 0.15 / 0.05: voting for the second choice in plurality gives 0.87, voting for the last gives 0.47. (Earlier versions reported the complement as "cognitive dissonance".)

## Ballots and sincere voting

| Ballot | Form | Sincere ballot |
|---|---|---|
| one name | one candidate | the favourite |
| ranking | full strict ranking | by utility |
| approval | approve or not | every candidate above the voter's mean utility (at least the favourite) |
| 0–5 | integer score per candidate | stretched to the voter's own utility range, rounded |
| ranking + approval line | ranking, the top k approved | ranking by utility, line below the last candidate above the mean |
| veto | one candidate against | the least liked |
| ranking, ties allowed | levels, equal = tied | strict ranking by utility (ties only arise from learning) |
| truncated ranking | the top k ranked, the rest unranked (tied last) | rank every candidate above the mean |

## Methods

| Method | Ballot | Rule |
|---|---|---|
| Random ballot | one name | a random voter's choice |
| Random pair runoff | ranking | two random ballots; same first choice wins, else majority runoff between the two |
| Plurality | one name | most first choices |
| Top-two runoff | ranking | top two by first choices, then majority runoff |
| IRV | ranking | eliminate fewest first choices among remaining |
| Borda | ranking | points by position |
| Baldwin | ranking | eliminate lowest Borda score, recounted among remaining |
| BTR-Borda | ranking | order by Borda score, bottom-two pairwise runoffs climbing up |
| Copeland | ranking | most pairwise wins (ties count half) |
| Minimax | ranking | smallest worst pairwise defeat (by margin) |
| Approval | approval | most approvals; sincere: above the voter's mean |
| Approval runoff | ranking + approval line | top two by approvals, majority runoff by ranking; a runoff tie goes to more approvals |
| Score | 0–5 | highest total; sincere: stretched to the voter's range |
| MJ | 0–5 | majority judgment: highest median grade; ties broken by repeatedly removing the median |
| STAR | 0–5 | top two by total, majority runoff; a runoff tie goes to the higher total |
| BTR-score | 0–5 | order by total score, bottom-two pairwise runoffs climbing up |
| MARS | 0–5 | as BTR-score; each match adds share preferring to score as share of maximum |
| ARA | 0–5 | ratings 3–5 are approvals; order by approvals; each match adds voters preferring to approvals (formerly MARA) |
| ARA-Top | 0–5 | as ARA, but only a 5 counts as approval |
| Anti-plurality | veto (one name against) | fewest vetoes wins; sincere: against the least liked |
| TRASH | ranking | "This Random Acronym Sounds Honest": the plurality runner-up and the anti-plurality runner-up (second fewest last places) meet in a runoff, and the one ranked lower by more voters wins; if both are the same candidate, it wins. A deliberately bad reference |
| Ranked pairs | ranking | lock pairwise wins from the largest margin down, skipping any that would close a cycle; the candidate with no locked defeat wins. Equal margins are locked in every order and averaged (at most 240 orders, otherwise 240 fixed random ones) |
| Schulze | ranking | strongest path by margins; a candidate whose path to every rival is at least as strong as the rival's path back wins |
| Approval runoff if needed | ranking + approval line | as Approval runoff, but a candidate approved by more than half the voters wins outright |
| Reciprocal score | 0–5 | interpretation of [electowiki](https://electowiki.org/wiki/Reciprocal_Score_Voting): a voter's faction is their top-rated candidate(s). R(X→Y) is the mean rating supporters of X give Y, as a share of 5. A score for c is multiplied by min(1, R(c→F)/R(F→c)), F the voter's faction: support that c's supporters do not return is cut. Scores for the voter's own top and for candidates nobody top-rates stay unchanged; highest total wins |
| Score+ | 0–5 | simplified variant from the [votingtheory.org thread](https://www.votingtheory.org/forum/topic/138/score-interval-score-with-additional-protection-against-the-chicken-dilemma): A defeats B robustly if A's total minus the scores both share (per voter the lower of the two) still exceeds B's total. Robustly defeated candidates are eliminated, each voter's scores among the rest are stretched again to 0–5, highest total wins |
| Condorcet-random-drop | ranking | Condorcet winner if there is one; otherwise voters are dropped one at a time in random order until the rest have one. Averaged over 32 fixed random orders (the same on every call), so probabilities come in steps of 1/32 |
| SICT | ranking, ties allowed | Symmetrical ICT: p(X,Y) if (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top); X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins; none or all unbeaten: most first places; otherwise most first places among the unbeaten. Sincere ballots are strict; ties are learned |
| MDDA | truncated ranking (ranked = approved) | disqualify every candidate that more than half the voters rank some other candidate above (unless all would be); most approved remaining wins. Sincere: rank everyone above the mean |
| BTR-IRV | ranking | each round the two with the fewest first choices meet pairwise; the loser is eliminated |
| Benham | ranking | IRV, but a Condorcet winner among the remaining candidates wins at once |

**Groups** for `--methods` and `--exclude` (a method can be in several):

| Group | Methods |
|---|---|
| `condorcet` | Baldwin, Copeland, Minimax, Ranked pairs, Schulze, SICT, Condorcet-random-drop, BTR-Borda, BTR-IRV, Benham |
| `plurality-based` | Plurality, Top-two runoff, IRV, BTR-IRV, Benham |
| `rated` | Score, MJ, STAR, BTR-score, MARS, ARA, ARA-Top, Reciprocal score, Score+ |
| `approval-based` | Approval, Approval runoff, Approval runoff if needed, MDDA |
| `references` | Random ballot, Random pair runoff, Anti-plurality, TRASH |

Borda is in no group. Example: `--methods condorcet STAR` or `--exclude references`.

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.

## Analysis

`analyze.py` reads one or more elections CSVs. Rows with the same parameters except the seed form one **condition**; seeds are pooled as replications.

**Uncertainty: paired bootstrap.** All methods face the same electorates, so each bootstrap replicate redraws whole elections with replacement and recomputes every method on that same draw (1000 replicates by default). This keeps comparisons paired: two methods can differ clearly even when the interval of each value on its own is wide, because elections differ much more from each other than methods differ within an election.

**What is reported per condition and method:**

| Output | Meaning |
|---|---|
| PE, WM, PE*, WM*, CE, CE*, sincerity | point estimate and 95% interval of each value on its own |
| dominated by | methods clearly better on at least one of the four axes and clearly worse on none, each with the axes on which it is better; the report shows at most three, the CSV all. Clearly: the paired 95% interval of the difference excludes 0, smaller differences count as ties. Without this, methods level on one axis (e.g. Condorcet methods, which mostly elect the same candidate with honest ballots) would be kept apart by noise-sized differences, and PE* and WM* would never get to decide between them |
| paired differences (`--compare "A:B"`) | difference A − B per axis with a 95% interval; marked as a tie when the interval includes 0 |

**How to interpret:**
- The values are the result. Whether a method lies on the Pareto front is already interpretation: it depends on which axes count and on which methods are included. "Dominated by" is only a pointer to where to look.
- The axes are kept separate on purpose. Each carries its own value judgement (utilitarian or majoritarian, honest or strategic); no weighting into a single score.
- Read results **per condition first**. The strategic axes depend strongly on the group size (see Coordination).
- An ordering between two methods only counts if their paired difference is not a tie.

**Plots** (in `--out`):

| File | Shows |
|---|---|
| `arrows-<c>.png` | PE/WM plane, every method's shift from honest (open) to learned ballots (filled, labelled) in one picture; filled points coloured by sincerity (red ramp, darker = less honest, capped at the 10th percentile of the plotted methods) |
| `shift-<c>.png` | the same shifts as one row per method, PE and WM panels; easier to read per method |
| `sincerity-<c>.png` | PE* and WM* against sincerity |
| `axes-heatmap.png` | the four axes, methods × conditions (when there are several) |

Methods are identified by labels, not colour. Random ballot, Random pair runoff, Anti-plurality and TRASH are left out of the scatter plots by default (`--hide`), since their distance compresses the scale; they stay in tables and heatmaps. `--hide` with no names shows all.

Sample size: with 160 elections per condition the intervals of single values are about ±3 points and many paired differences are still ties. For conclusions, several hundred elections per condition and at least two seeds are advisable.
