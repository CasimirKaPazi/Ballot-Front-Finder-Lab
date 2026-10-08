# Voting Pareto Lab

Python 3 + NumPy; the analysis also needs matplotlib. Honest and learned-strategy elections, evaluated on a Pareto front.

This file describes the software: the model, its parameters, the methods and the analysis. What the simulations have shown so far, with the commands that produced it, is in [FINDINGS.md](FINDINGS.md).

```
python test_methods.py                                   # hand-built checks
python run.py                                            # one condition at the defaults, prints a table
python sweep.py --group-size 10 1 0 --seeds 1 2          # grid of conditions -> sweep-elections.csv
python analyze.py sweep-elections.csv --compare "BTR-score:STAR"   # -> output/analysis/
```

All outputs go into the folder `output/` (created when needed): an `--out` name without a directory is placed there, and `analyze.py` looks there for CSVs given without a directory. An `--out` with a directory (`--out runs/test`) is used as given.

200 elections with all 42 methods at the default settings take about 14 minutes on 2 cores; time scales linearly with cores. More candidates, more voters and smaller groups take longer. Use `--methods` or `--exclude` to run fewer, by name or by group (see Methods).

## Files

| File | Contents |
|---|---|
| `scenario.py` | Electorate generator: spatial clusters, Zipf dimensions, absolute utilities `exp(-distance / width)` |
| `methods.py` | Ballot types, sincere ballots, exact tallies on ballot matrices (ties shared), method groups |
| `strategy.py` | Learning engine, sincerity and strategy types |
| `nomination.py` | Strategic nomination: exits, re-entry and clones from the initial field |
| `metrics.py` | Efficiencies for three baselines, winner's worst margin, Condorcet efficiency, electorate properties, Pareto front |
| `run.py` | Tallies every electorate with sincere and learned ballots, runs in parallel, writes CSV with the run's parameters as columns |
| `sweep.py` | Runs a grid of conditions (or one parameter at a time) and seeds into one CSV |
| `analyze.py` | Paired bootstrap, intervals, dominance pointers, paired comparisons, rank stability, Markdown report |
| `plots.py` | The analysis plots |
| `test_methods.py` | Hand-built checks |
| `FINDINGS.md` | Results so far and their interpretation |

## Strategy engine

- **Groups:** strategic voters with the same sincere ranking, split at random into subgroups of at most `--group-size` voters (see Coordination below). A group either votes sincerely (each member their own sincere ballot) or casts one shared ballot.
- **Polls:** each round draws 10 bootstrap resamples of the electorate. A proposal is judged by the group's total expected utility averaged over these samples, so a group gains in proportion to how often it would be pivotal, not only when it is decisive in the exact electorate.
- **Proposals:** one small change to the group's ballot:
  - one name: switch to another candidate
  - ranking: swap two candidates
  - approval: toggle one candidate (never empty or full)
  - ranking with approval line: swap two candidates, or move the line up or down by one
  - veto: move the veto to another candidate
  - ranking with ties: move one candidate to any other level (equal levels are ties)
  - truncated ranking: swap two candidates, or rank one more or one fewer
  - score: set one candidate to a different value 0–max. For ranges above 0–5, half of the proposals use the six levels of the 0–5 scale stretched to the range (0, 20, … 100 for 0–100), so the chance of trying an extreme does not shrink as the range grows and learning is not slowed by the range alone
  - three grades: set one candidate to a different grade

  Voters start sincere. A group that has not deviated yet changes the sincere ballot of its average member (mean utilities of the group); a deviating group changes its current shared ballot, and sometimes (15% of proposals) proposes going back to sincere.
- **Acceptance:** a proposal is kept only if it raises the group's expected utility. A return to sincere is also kept when it is not worse: voters prefer honesty when indifferent. Without this, a deviation that once paid off stays forever once it stops mattering, and sincerity keeps falling with the number of rounds instead of reflecting what strategy actually pays.
- **Rounds:** 20, groups in random order each round; no early stop. Each group tries only one random change per round, so a round without any accepted change does not mean learning has finished, and `settled` (no change in the last round) is a weak sign of convergence. The tests behind both choices are in FINDINGS.md.
- **Feature sums:** every tally except Condorcet-random-drop depends on the ballots only through sums over voters of per-voter features (totals, first and last places, the pairwise matrix, grade counts, first choices among each subset of candidates, ...). Each method lists its features and decides from their sums. During learning the sums are kept per poll sample and updated only for the group that changes. All features are integers, so this gives exactly the same learned ballots as recounting (checked against the earlier code).
- **Conditional top-two runoff** is not a separate method: a candidate with more than half the first choices also wins every runoff, since the runoff reuses the same rankings. (In a real two-round election voters vote again, with new information and different turnout; that is not modelled.)
- **Random streams:** each method learns with its own random stream per election, keyed by the method's name, so results do not change when methods are added or reordered.

## Electorates

Each election draws a fresh electorate (`scenario.py`):

- cluster centres: `--clusters` points, normal in `--dims` dimensions;
- cluster sizes: random weights from a flat Dirichlet distribution, so some elections have one dominant cluster and others are balanced;
- voters: each assigned to a cluster by these weights, placed normally around its centre with standard deviation `--spread` (per dimension, before the Zipf scaling);
- candidates: drawn from the same distribution as voters (cluster by the same weights, same spread);
- utilities: `u = exp(-distance / width)`, absolute and not normalised: 1 at zero distance, towards 0 far away. A smaller `--width` makes voters care only about nearby candidates.

**Zipf dimensions** (the default, `--zipf 1`), after Clay Smith's Mixture of Zipf Gaussians ([article](https://cdsmithus.medium.com/simulating-elections-with-spatial-voter-models-1ff50892390), [code](https://github.com/cdsmith/spatial-voting), `src/Model.hs`): dimension i (1, 2, 3, ...) has standard deviation 1/i — for cluster centres, voters and candidates alike — so one issue dominates and the others matter less and less, without being absent.

- Scaling follows Smith's code: the standard deviation is 1/i, so the variance is 1/i². (The article's wording "proportional to 1/n" could also be read as variance 1/n; the code is unambiguous.) The total variance then converges (to π²/6), so beyond about 4 dimensions their number hardly matters. The default of 10 dimensions is well past that.
- Each cluster applies the 1/i scaling to its own slightly shuffled order of dimensions (each rank perturbed by normal noise with standard deviation 0.1 × dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues they vary on most. Cluster centres keep the global order.
- Taken over from Smith: the 1/i scaling and the per-cluster shuffle. Not taken over: his other settings (10 clusters with uniform random weights, cluster centres uniform in ±3/i, a random spread factor per cluster, utility = −distance); our Dirichlet weights, normal centres, one spread and `exp(-distance / width)` stay.
- `--zipf 0` gives the earlier model: all dimensions with standard deviation 1 (then use few dimensions, e.g. 2). Distances are longer there, so the same width gives steeper preferences.

## Parameters

`run.py` (one condition). The defaults are the base condition of the latest sweeps (see FINDINGS.md):

| Flag | Default | Meaning |
|---|---|---|
| `--elections` | 200 | electorates |
| `--voters` | 99 | voters per election (odd, so pairwise majorities rarely tie) |
| `--cands` | 6 | candidates |
| `--dims` | 10 | issue-space dimensions |
| `--zipf` | 1 | 1: dimension i has scale 1/i (Zipf dimensions, above); 0: all dimensions equal |
| `--clusters` | 3 | voter clusters |
| `--spread` | 0.5 | standard deviation of voters around their cluster centre (centres have 1), before the Zipf scaling |
| `--width` | 1.5 | distance scale of the utility fall-off |
| `--seed` | 1 | random seed; election e uses the stream (seed, e), so all methods face the same electorates |
| `--rounds` | 20 | learning rounds |
| `--samples` | 10 | poll resamples per round |
| `--group-size` | 10 | most voters that coordinate as one group: a number of voters (1 = every voter alone), a share of the electorate between 0 and 1 (0.05 = 5% of the voters, rounded, at least 1), or 0 = whole ranking group |
| `--strategic` | 0.5 | share of voters who vote strategically (see Partial strategy) |
| `--max-score` | 5 | score ballots range over 0–max (see Score range) |
| `--exits` | 0 | 1: candidates may withdraw and re-enter strategically (see Strategic nomination) |
| `--clones` | 0 | most clones in the race at once, fielded strategically; 0 = none (see Strategic nomination) |
| `--workers` | all cores | parallel processes |
| `--methods` | all | run only these methods (start from none); method or group names (see Methods), case ignored |
| `--exclude` | none | run all methods except these |
| `--out` | `results` | output prefix: `output/<out>-elections.csv` and `output/<out>-summary.csv` |

`sweep.py` takes the same flags, but every condition flag (`--voters` … `--width`, `--zipf`, `--rounds`, `--samples`, `--group-size`, `--strategic`, `--max-score`, `--exits`, `--clones`) accepts several values, and `--seeds` several seeds; it runs the full grid and writes `output/<out>-elections.csv` (default `--out sweep`), one run after another. With `--one-at-a-time`, the first value of every flag is the base, and each other value is run with everything else at the base, instead of the full grid.

`analyze.py`:

| Flag | Default | Meaning |
|---|---|---|
| (positional) | | one or more elections CSVs |
| `--out` | `analysis` | output directory (`output/analysis`) |
| `--boot` | 1000 | bootstrap replicates |
| `--compare` | none | pairs `"A:B"` for paired differences |
| `--hide` | Random pair runoff, Anti-plurality, TRASH | methods left out of the scatter and sweep plots; `--hide` alone shows all |
| `--show` | all | only these methods or groups in the scatter and sweep plots (overrides `--hide`), e.g. `--show condorcet STAR` |
| `--seed` | 0 | bootstrap random seed |

## Robustness: one parameter at a time

To see how much a model assumption matters, vary each parameter on its own around a base condition and check whether the methods keep their order:

```
python sweep.py --one-at-a-time --clusters 3 5 --spread 0.5 0.75 --width 1.5 1.0 \
    --group-size 10 1 0 --strategic 0.5 1 --seeds 1 2 --elections 400 --out model
python analyze.py model-elections.csv --out model-analysis
```

Conditions are labelled by how they differ from the base (`c1: base`, `clusters 5`, ...).

`analyze.py` adds a **rank stability** table when there are several conditions: for each axis, the rank correlation (Spearman) between the methods' values in a condition and in the base (the first condition). The methods hidden in the plots are left out. Its first row is a noise reference: the agreement expected from sampling noise alone between two runs of the base condition (the base's even and odd elections correlated, then scaled up to full size with the Spearman-Brown formula `2r / (1 + r)`). A condition near or above that row orders methods as the base does; a value clearly below it means the parameter changes which methods do well. With few elections everything is noise, and VSE* in particular needs many elections (several hundred per condition): strategic differences between methods are small. Many methods lie close together on each axis, so rank correlation is a strict test; the heatmap shows where the changes come from.

An **electorates** table (and a line in each condition's section) describes the generated electorates themselves, independent of the methods: how often a Condorcet winner exists, how often it also has the highest total utility, how much VSE electing it loses (ratio of sums, as VSE), and how steep preferences are (u(favourite) ÷ u(least liked), median over voters, then median over elections). The third column explains most VSE differences between Condorcet and rated methods; the last one is a concrete handle for choosing `--width`.

## Coordination: what the strategic axes measure

How many voters move together decides which strategies pay, and the methods differ sharply in this.

- **Compromise** (voting for a viable candidate instead of a hopeless favourite) pays even for a single voter. It is the main strategy in TTR (top-two runoff) and IRV, and it tends to carry the Condorcet winner into the final round.
- **Burial** (ranking or rating a rival below a candidate one likes less, to create a cycle or to drop the rival in a runoff) only pays when many voters do it together. A single voter out of 99 is almost never pivotal for it. It is the main strategy against Condorcet methods, STAR and BTR-score, and it targets exactly the Condorcet winner.

With `--group-size 0` (whole ranking groups) the strategic axes measure resistance to **organised factional strategy**: every voter with the same ranking follows one recommendation, the case most favourable to burial. With `--group-size 1` they measure resistance to **individual strategy**, where burial almost never pays. The default of 10 lies in between; results under strategy should always be read together with the group size used.

Groups are formed from voters with the same full sincere ranking. With many candidates there are many possible rankings (720 for 6 candidates), so most ranking groups are small whatever the cap, and organised blocs become rare: with 6 candidates and 99 voters, the group size changes little. Coordination by favourite or by cluster would be a different model.

A second, smaller asymmetry: learning is myopic. A burying group does not anticipate that the Condorcet winner's supporters may bury back, or that the cycle it creates can elect a candidate it likes even less (the chicken dilemma). This probably makes burial look somewhat more attractive than it would be to far-sighted voters.

Smaller groups mean more groups, and run time grows with their number.

## Partial strategy

`--strategic p` makes a random share p of the voters strategic; the others always vote sincerely. Each voter is strategic with probability p, drawn from its own random stream per election, so the same voters are strategic for every method. Groups (and their size cap) form only among the strategic voters. p = 1 gives full strategy; p = 0 gives the sincere results. The default 0.5 is an intermediate level, in the spirit of other simulations that report 0, 50 and 100% strategy. Sincerity and the strategy types are averaged over all voters, including the sincere ones.

Run time falls less than in proportion to p: halving the strategic voters removes fewer than half of the groups, since many groups are small.

## Score range

`--max-score m` sets the range of the score ballot to 0–m for every method that uses it (Score, MJ, STAR, BTR-score, MARS, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score). The default 0–5 is an implementation choice, not part of these methods; other simulations (e.g. Quinn's VSE) also run 0–10 and 0–100. With a coarse range, sincere voters must give candidates they like differently the same score, so pairwise preferences are lost: this is the cap on WME seen for the rated and hybrid methods. A wider range reduces it. MCA and 321 keep their three grades, and methods on other ballots are unaffected; the electorates are the same for every value, so a sweep (`sweep.py --max-score 5 10 100`) isolates the effect of the range, and its chain plot draws the reachable frontier.

Rules that refer to the range scale with it: ARA approves ratings above the middle, ARA-Top only the top score, Score+ stretches to the full range, Reciprocal score measures support as a share of the top score. Learning: see the score mutation in Strategy engine.

## Strategic nomination

`--exits 1` and `--clones N` let the candidates of the initial draw change the field before the vote (`nomination.py`; after Green-Armytage 2014, "Strategic voting and nomination", *Social Choice and Welfare* 42, with clones added). Nobody new enters.

- **Moves.** Exit (at least two original candidates stay in), re-entry after an exit, fielding a clone of any original in the race (itself included; at most one clone per original and N in the race), withdrawing a clone it fielded. A clone is a copy at a small random offset (0.01 per dimension), so voters rank it next to its original and split between the two.
- **Who moves and why.** Candidates sit in the issue space and value a winner like a voter at their position (exp(−distance/width); themselves 1). A clone counts exactly as its original. A candidate makes a move only if it strictly improves its expected utility under the method's result with **sincere** ballots on the new field: candidates foresee honest voting, not learned strategy.
- **Dynamics.** Rounds in which the candidates take turns in random order, each making its best improving move. A round without a move ends it (an equilibrium); otherwise it stops after 20 rounds. Each method gets its own field. The voters then vote on that field, honestly and with learned ballots as usual.
- **Baseline.** Results are judged on the **original field**: a clone's chance of winning counts as its original's, and withdrawn candidates stay in the comparison. VSE, WME, the random ballot, the best candidate and the reachable frontier are therefore the same as without nomination, and the change in VSE and WME is what nomination does to the outcome. Withdrawing the best candidate counts as a loss.
- **Report.** With nomination on, each condition gets a table per method: how often the honest result changed (Green-Armytage's measure of vulnerability, here after the dynamics rather than for a single move), the average number of exits and clones at the end, and how often the field settled. A sweep `--exits 0 1` or `--clones 0 2` draws chain plots with the frontier.

Cost: each candidate move is checked with one sincere tally on the changed field, a few hundred tallies per method and election; learning remains the larger part.

## Axes and other outputs

Both axes are **efficiencies with the same two reference points**, as ratios of sums over all elections:

`E = Σ(X_method − X_rb) / Σ(X_best − X_rb)`

- **0 = random ballot**: the winner is the favourite of a randomly drawn voter. It needs no assumption about which candidates stand, and every reasonable method should beat it.
- **100 = the best candidate** of each election. Values above 100 cannot be reached; the plots grey that area out. Values below 0 are worse than random ballot.
- The zero is a common level for all elections; it does not judge individual results. Compromise candidates are therefore still rewarded: in an election where the best candidate is nobody's favourite, electing it counts fully toward the numerator.

**VSE** (voter satisfaction efficiency), first axis: X = total utility of the winner, `U(c) = Σ_v u_v(c)`; best = the candidate with the highest total utility. The usual VSE puts the zero at a random candidate instead. With candidates drawn from the same distribution as voters the two differ only in scale, so the order of methods is the same. No utility normalisation: a voter who cares little (all candidates far away) counts little.

**WME** (worst-margin efficiency), second axis, a refined Condorcet measure: X = the winner's worst margin, its smallest head-to-head margin against any other candidate, by sincere preferences, as a share of voters. Best = the candidate with the best worst margin.

- The motivation is legitimacy: after the election voters judge the result, and if someone other than the Condorcet winner won, they can argue that a majority would have preferred another candidate. The worst margin measures how strong that argument is, instead of only whether it exists: losing a Condorcet winner by one vote costs little, electing a candidate whom a large majority prefers a rival to costs a lot.
- Every election counts, cycles included, and the value does not depend on the number of voters.
- With honest ballots Minimax reaches 100 by construction (it elects the candidate with the best worst margin), and all Condorcet methods reach it whenever there is a Condorcet winner, just as they reach 100 on CE. The strategic version WME* is where methods are tested.

**Reachable frontier.** VSE and WME pull apart: the candidate with the highest total utility is often not the one with the best worst margin, so no rule can reach 100 on both. `analyze.py` computes what any rule could reach in the simulated elections if it knew every voter's sincere utilities: for each weight λ between 0 and 1 it elects, in every election, the candidate with the highest λ × (its VSE contribution) + (1 − λ) × (its WME contribution). Since both axes are ratios of sums with denominators that do not depend on the rule, this traces the exact boundary of the reachable region (lotteries fill in between the points). The plots draw it as a curve and grey out the region beyond it; the report gives its two ends and its most balanced point (highest VSE + WME). A method's distance to the curve is what it loses to having only ballots, not utilities, and to strategy. For this, run.py stores every candidate's total utility and worst margin per election (columns U_cands, WM_cands).

**Condorcet efficiency (CE)**, reported alongside but not an axis: among elections with a Condorcet winner by sincere utilities, the probability the method elects it. It ignores elections without a Condorcet winner and does not distinguish narrow from clear ones.

VSE, WME and CE use sincere ballots; VSE*, WME* and CE* (in the CSV: VSE_learned, WME_learned, CE_learned) the learned ballots. In all cases the preferences that count are the sincere ones.

Earlier versions (before 2026-10-06) used PE, with a utility-proportional lottery as zero, and WM, the raw worst margin in % of voters; FINDINGS.md says which applies where.

**Output files of `run.py`:** `<out>-elections.csv` has one row per (election, method, ballots = sincere or learned) with the run's parameters as columns; the quantities for the axes: U_method, U_best, U_rb (random ballot), U_rand (random candidate, for the usual VSE), WM (the winner's worst margin, share of voters), WM_best, WM_rb; has_cw and p_cw (probability of electing the Condorcet winner); the electorate properties cw_is_util, U_cw and fav_ratio (the same in every row of an election; see Robustness); sincerity, settled and the strategy types. `<out>-summary.csv` and the printed table have one row per method:

| Column | Meaning |
|---|---|
| VSE, WME | voter satisfaction and worst-margin efficiency, sincere ballots |
| VSE_learned, WME_learned | the same with learned ballots |
| CE, CE_learned | Condorcet efficiency, sincere and learned ballots |
| sincerity | mean sincerity of the learned ballots, over all voters (below) |
| settled | share of elections where no group changed its ballot in the last round; cycles never settle |

`on_front` in the summary is a quick check on point estimates (VSE, WME, VSE_learned, WME_learned, no uncertainty). The analysis uses the tie-aware "Dominated by" instead (see Analysis).

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
| score 0–max | integer score per candidate (`--max-score`, default 5) | stretched to the voter's own utility range, rounded |
| ranking + approval line | ranking, the top k approved | ranking by utility, line below the last candidate above the mean |
| veto | one candidate against | the least liked |
| ranking, ties allowed | levels, equal = tied | strict ranking by utility (ties only arise from learning) |
| truncated ranking | the top k ranked, the rest unranked (tied last) | rank every candidate above the mean |
| three grades | good / OK / bad per candidate | stretched to the voter's own utility range, rounded to 0–2 (as the score ballot, with three levels; not affected by `--max-score`) |

## Methods

"0–5" in the Ballot column is the score ballot with the default range; `--max-score` changes it for all of these methods at once.

| Method | Ballot | Rule |
|---|---|---|
| Random pair runoff | ranking | two random ballots; same first choice wins, else majority runoff between the two |
| Plurality | one name | most first choices |
| TTR | ranking | top two by first choices, then majority runoff |
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
| ARA | 0–5 | ratings above the middle of the range are approvals (0–5: 3–5; 0–10: 6–10); order by approvals; each match adds voters preferring to approvals (formerly MARA) |
| ARA-Top | 0–5 | as ARA, but only the top score counts as approval |
| Anti-plurality | veto (one name against) | fewest vetoes wins; sincere: against the least liked |
| TRASH | ranking | "This Random Acronym Sounds Honest": the plurality runner-up and the anti-plurality runner-up (second fewest last places) meet in a runoff, and the one ranked lower by more voters wins; if both are the same candidate, it wins. A deliberately bad reference |
| Ranked pairs | ranking | lock pairwise wins from the largest margin down, skipping any that would close a cycle; the candidate with no locked defeat wins. Equal margins are locked in every order and averaged (at most 240 orders, otherwise 240 fixed random ones) |
| Schulze | ranking | strongest path by margins; a candidate whose path to every rival is at least as strong as the rival's path back wins |
| Approval cond. runoff | ranking + approval line | as Approval runoff, but a candidate approved by more than half the voters wins outright |
| Reciprocal score | 0–5 | interpretation of [electowiki](https://electowiki.org/wiki/Reciprocal_Score_Voting): a voter's faction is their top-rated candidate(s). R(X→Y) is the mean rating supporters of X give Y, as a share of the top score. A score for c is multiplied by min(1, R(c→F)/R(F→c)), F the voter's faction: support that c's supporters do not return is cut. Scores for the voter's own top and for candidates nobody top-rates stay unchanged; highest total wins |
| Score+ | 0–5 | simplified variant from the [votingtheory.org thread](https://www.votingtheory.org/forum/topic/138/score-interval-score-with-additional-protection-against-the-chicken-dilemma): A defeats B robustly if A's total minus the scores both share (per voter the lower of the two) still exceeds B's total. Robustly defeated candidates are eliminated, each voter's scores among the rest are stretched again to the full range, highest total wins |
| Condorcet-random-drop | ranking | Condorcet winner if there is one; otherwise voters are dropped one at a time in random order until the rest have one. Averaged over 32 fixed random orders (the same on every call), so probabilities come in steps of 1/32 |
| SICT | ranking, ties allowed | Symmetrical ICT: p(X,Y) if (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top); X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins; none or all unbeaten: most first places; otherwise most first places among the unbeaten. Sincere ballots are strict; ties are learned |
| MDDA | truncated ranking (ranked = approved) | disqualify every candidate that more than half the voters rank some other candidate above (unless all would be); most approved remaining wins. Sincere: rank everyone above the mean |
| BTR-IRV | ranking | each round the two with the fewest first choices meet pairwise; the loser is eliminated |
| Benham | ranking | IRV, but a Condorcet winner among the remaining candidates wins at once |
| Black | ranking | Condorcet winner if there is one, otherwise the Borda winner |
| Ranked Robin | ranking, ties allowed | Equal Vote's Condorcet method: most pairwise wins; ties go to the greatest sum of margins against the other tied candidates, then against all candidates; remaining ties (its 3rd and 4th tiebreakers) are split. Sincere ballots are strict; ties are learned |
| Nanson | ranking | repeatedly eliminate every candidate whose Borda score among the remaining is at or below the average (Baldwin eliminates only the lowest) |
| MMWV | ranking, ties allowed | Minimax by winning votes: the worst defeat is measured by the votes for the winner of that pair. On strict rankings identical to Minimax (margins); differs only once equal rankings are learned |
| MMPO | ranking, ties allowed | minimax pairwise opposition: the smallest largest number of voters ranking some rival above the candidate, whether or not that rival wins the pair. Not a Condorcet method; on strict rankings identical to Minimax |
| MMPOWV | ranking, ties allowed | proposed in this lab: (1 − largest pairwise opposition / n) × (1 − worst defeat in winning votes / n); the highest product wins. Multiplying makes a weakness on either costly and is smooth, without thresholds for strategy to aim at. On strict rankings it orders candidates like Minimax, so it equals Minimax with honest ballots and differs only once equal rankings are learned. Not Condorcet-consistent once ties are used, so in no group |
| Smith//Score | 0–5 | highest score total among the Smith set (pairwise from the ratings, equal ratings no preference). With a Condorcet winner by the ballots the same as BTR-score; they differ only in cycles |
| ER-IRV | ranking, ties allowed | IRV where every candidate tied at the top of a ballot (among the remaining) gets a whole vote, as in approval. Sincere ballots are strict, so it equals IRV with honest ballots; ties are learned |
| Supplementary vote | ranking (first two used) | top two by first preferences; a ballot whose first preference is out counts for its second preference if that is a finalist, otherwise it is exhausted |
| Dowdall | ranking | positional, 1, 1/2, 1/3, … points by rank (Nauru) |
| MCA | three grades | majority choice approval, MCA-A variant: the first grade level (good, then good or OK, then all) at which some candidate is rated that high by more than half the voters decides; one such candidate wins, among several the most approved (good or OK) |
| 321 | three grades | 3-2-1 voting: 3 semifinalists with the most "good" (the third only with at least half as many as the first), of them 2 finalists with the fewest "bad", then the finalist rated above the other on more ballots. Ties: higher score (good 2, OK 1), then split. The party rule for semifinalists does not apply |

**Groups** for `--methods` and `--exclude` (a method can be in several):

| Group | Methods |
|---|---|
| `condorcet` | Baldwin, Nanson, Copeland, Ranked Robin, Black, Minimax, MMWV, Ranked pairs, Schulze, SICT, Condorcet-random-drop, BTR-Borda, BTR-IRV, Benham, Smith//Score |
| `plurality-based` | Plurality, TTR, Supplementary vote, IRV, ER-IRV, BTR-IRV, Benham |
| `rated` | Score, MJ, MCA, Reciprocal score |
| `hybrid` | STAR, BTR-score, MARS, ARA, ARA-Top, Score+, Smith//Score, 321 (ratings combined with pairwise comparisons or eliminations) |
| `approval-based` | Approval, Approval runoff, Approval cond. runoff, MDDA |
| `positional` | Borda, Dowdall |
| `references` | Random pair runoff, Anti-plurality, TRASH |

Example: `--methods condorcet STAR` or `--exclude references`.

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.

## Analysis

`analyze.py` reads one or more elections CSVs. Rows with the same parameters except the seed form one **condition**; seeds are pooled as replications.

**Uncertainty: paired bootstrap.** All methods face the same electorates, so each bootstrap replicate redraws whole elections with replacement and recomputes every method on that same draw (1000 replicates by default). This keeps comparisons paired: two methods can differ clearly even when the interval of each value on its own is wide, because elections differ much more from each other than methods differ within an election.

**What is reported per condition and method:**

| Output | Meaning |
|---|---|
| VSE, WME, VSE*, WME*, CE, CE*, sincerity | point estimate and 95% interval of each value on its own |
| dominated by | methods clearly better on at least one of the four axes and clearly worse on none, each with the axes on which it is better; the report shows at most three, the CSV all. Clearly: the paired 95% interval of the difference excludes 0, smaller differences count as ties. Without this, methods level on one axis (e.g. Condorcet methods, which mostly elect the same candidate with honest ballots) would be kept apart by noise-sized differences, and VSE* and WME* would never get to decide between them |
| paired differences (`--compare "A:B"`) | difference A − B per axis with a 95% interval; marked as a tie when the interval includes 0 |

**How to interpret:**
- The values are the result. Whether a method lies on the Pareto front is already interpretation: it depends on which axes count and on which methods are included. "Dominated by" is only a pointer to where to look.
- The axes are kept separate on purpose. Each carries its own value judgement (utilitarian or majoritarian, honest or strategic); no weighting into a single score.
- Read results **per condition first**. The strategic axes depend strongly on the group size (see Coordination).
- An ordering between two methods only counts if their paired difference is not a tie.

**Plots** (in `--out`):

| File | Shows |
|---|---|
| `arrows-<c>.png` | VSE/WME plane (the region beyond the reachable frontier greyed out), every method's shift from honest (open) to learned ballots (filled, labelled) in one picture; filled points coloured by sincerity (red ramp, darker = less honest, capped at the 10th percentile of the plotted methods) |
| `shift-<c>.png` | the same shifts as one row per method, VSE and WME panels; easier to read per method |
| `sincerity-<c>.png` | VSE* and WME* against sincerity |
| `axes-heatmap.png` | the four axes, methods × conditions (when there are several) |
| `chain-<parameter>.png` | sweeps: for each parameter varied around the base (the first condition), each method's learned values as a chain of arrows through the parameter's values, light to dark; one panel per method group, all on the same scale. If the parameter only affects learning (strategic share, group size, rounds, poll samples), the electorates are the same and each chain starts at the method's honest point; otherwise the honest values form their own dashed chain |

Sweep plots appear automatically for every parameter that has at least two conditions differing from the base only in that parameter: a one-at-a-time sweep (`sweep.py --one-at-a-time`) or a grid over a single parameter. They show point estimates only; with few elections the zigzags in a chain are mostly noise. One far-off method (often Plurality) sets the shared scale for all panels; `--hide` or `--show` zooms in.

Methods are identified by labels, not colour. Random pair runoff, Anti-plurality and TRASH are left out of the scatter plots by default (`--hide`), since their distance compresses the scale; they stay in tables and heatmaps. `--hide` with no names shows all.

Sample size: with 160 elections per condition the intervals of single values are about ±3 points and many paired differences are still ties. For conclusions, several hundred elections per condition and at least two seeds are advisable.
