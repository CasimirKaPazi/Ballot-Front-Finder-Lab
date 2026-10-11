# Voting Pareto Lab

A simulator for comparing single-winner voting methods: plurality, runoffs, IRV, Condorcet methods, score and approval methods and their hybrids, 65 in all, plus eight test methods. It asks how good the winner of each method is, not only when everyone votes honestly but also when voters respond strategically to polls, and how honest voters can afford to be under each method.

**The question.** Every voting method turns ballots into a winner, and every method can be gamed to some degree. Comparisons with honest ballots alone flatter methods that are easy to manipulate; comparisons under fully rational strategy depend on assumptions nobody believes. This lab places voters in between: some of them read polls and adjust their ballots when it pays, the way people vote tactically in real elections, and it measures what the method makes of that.

**How a simulation runs.**

1. **An electorate.** Voters and candidates are points in an issue space (several dimensions of decreasing importance, voters in clusters). A voter likes a candidate less the farther away it is. Each election draws a new electorate (README: Electorates).
2. **Honest ballots.** Each voter fills in the method's ballot sincerely: a name, a ranking, approvals or scores, depending on the method (Ballots and sincere voting).
3. **Poll rounds.** Polls show the method's own counts from a sample of the electorate. Strategic voters (by default half the electorate), in groups with the same preferences, change their ballot if that improves the result for them in the polls; all groups answer the same poll at once, and the next poll shows the new ballots (Strategy engine).
4. **Optionally**, candidates may withdraw or field clones beforehand (Strategic nomination), and a diagnostic checks whether some voters would do better by staying home (Participation).

**What is measured.** Two efficiencies, both 0 for a random ballot and 100 for the best candidate of each election (Axes):

- **VSE**, voter satisfaction efficiency: the winner's total utility (how well the electorate as a whole likes it);
- **WME**, worst-margin efficiency: the winner's smallest head-to-head margin against any rival (how strongly a majority could object that another candidate should have won).

The two pull apart, so no method can reach 100 on both; the lab computes the boundary that any rule could reach knowing everyone's true preferences (the reachable frontier). Each value comes with sincere ballots (VSE, WME) and after the poll rounds (VSE*, WME*), together with the **sincerity** of the strategic ballots. The main comparison is on VSE*, WME* and sincerity: how good the result is with strategic voters, and how honest they can be.

**What you get.** `run.py` and `sweep.py` produce CSV files with one row per election, method and ballot set; `analyze.py` turns them into a Markdown report with paired bootstrap intervals, the methods no other method beats (the front), paired comparisons and plots: how strategy moves each method in the VSE/WME plane, how methods change across parameter sweeps, and where they stand relative to the frontier.

The model builds on voter satisfaction efficiency (Jameson Quinn's VSE simulations) and on Clay Smith's spatial voter model; in the strategy model, rounds of best responses to the previous poll (level-k reasoning), the leader rule of Myerson and Weber (1993) emerges instead of being coded. Its assumptions are spelled out in the sections below, and eight Inverse-* methods (honest voting elects the worst candidate) test whether voters learn their way back.

Python 3.9 or newer + NumPy; the analysis also needs matplotlib (`pip install -r requirements.txt`). This file describes the software: the model, its parameters, the methods and the analysis. What the simulations have shown so far, with the commands that produced it, is in [FINDINGS.md](FINDINGS.md).

```
python test_methods.py                                   # hand-built checks
python run.py                                            # one condition at the defaults, prints a table
python sweep.py --rounds 1 2 3 --aggressive 0 0.5 1    # grid of conditions -> sweep-elections.csv
python analyze.py sweep-elections.csv --compare "BTR-score:STAR"   # -> output/analysis/
```

All outputs go into the folder `output/` (created when needed): an `--out` name without a directory is placed there, and `analyze.py` looks there for CSVs given without a directory. An `--out` with a directory (`--out runs/test`) is used as given.

200 elections with all methods at the default settings (three poll rounds, 10 polls per round) took about 20 minutes in one measurement; time scales linearly with cores, with the number of poll rounds and with the polls per round. To try a new method, there is no need to rerun the others: see Combining runs. More candidates and more voters take longer. Use `--methods` or `--exclude` to run fewer, by name or by group (see Methods).

## Files

| File | Contents |
|---|---|
| `scenario.py` | Electorate generator: spatial clusters, Zipf dimensions, absolute utilities `exp(-distance / width)` |
| `methods.py` | Ballot types, sincere ballots, exact tallies on ballot matrices (ties shared), method groups |
| `strategy.py` | Learning engine, sincerity and strategy types |
| `nomination.py` | Strategic nomination: exits, re-entry and clones from the initial field |
| `participation.py` | Participation check (no-show paradox) |
| `metrics.py` | Efficiencies for three baselines, winner's worst margin, Condorcet efficiency, electorate properties, Pareto front |
| `run.py` | Tallies every electorate with sincere and learned ballots, runs in parallel, writes CSV with the run's parameters as columns |
| `sweep.py` | Runs a grid of conditions (or one parameter at a time) and seeds into one CSV |
| `analyze.py` | Paired bootstrap, intervals, dominance pointers, paired comparisons, rank stability, Markdown report |
| `plots.py` | The analysis plots |
| `test_methods.py` | Hand-built checks |
| `FINDINGS.md` | Results so far and their interpretation |

## Strategy engine

Two models of strategic voting (`strategy.py`). **Poll rounds** is the default; **convergent learning**, a long-run model kept for comparison, is available with `--converge N`.

### Poll rounds (default)

One election, with a few polls before it. Voters see only published poll results, never anyone's preferences, and answer them all at the same time.

- **Round 0:** everyone votes sincerely (naive, uninformed). `--rounds 0` reports this as the strategic result.
- **Round r** (1 … `--rounds`, default 3): a poll of round r−1 is published, and every strategic group answers it at once. Nobody sees anyone else's answer before the next poll. The strategic axes report the last round. Each round answers a pure poll of the round before (level-k reasoning: round 1 best-responds to sincere voters, round 2 to round-1 voters, …).
- **What a poll shows:** the method's own counts, the summed ballot features it decides from (first preferences for Plurality, approvals for Approval, the pairwise matrix for Condorcet methods, score totals and head-to-heads for STAR, …), from a random sample of `--poll-size` voters (default half the electorate, rounded to an odd number: 99 → 49, 100 → 51, 200 → 101), scaled up to the electorate. `--samples` independent polls (30) stand for the uncertainty. For some methods these counts say more than a real poll would (IRV's counts include every elimination order); methods with richer counts may profit from that.
- **Uncertainty:** voters treat a poll as `--poll-error` times (2) less precise than its sampling error: the polls are spread around the true count by that factor. Real polls miss results by about twice their stated sampling error (Shirani-Mehr, Rothschild, Goel & Gelman 2018). With the sampling error alone, the leader's supporters almost never see a poll in which they are behind, so they never protect their lead, while the runner-up's supporters attack it.
- **Groups:** whole ranking groups (strategic voters with the same sincere ranking, no cap). Voters who share a ranking and see the same poll reach the same answer, so they act as a bloc. Factions across rankings are not built in; they emerge when groups answer the same poll alike (all who prefer the runner-up to the leader, say).
- **Leaders:** the two candidates that win most often over the round's polls, the same for everyone. If one candidate wins every poll, the polls are widened (deviations × 2, 4, …; counts that would cross zero are clipped there) until another one wins somewhere: the race that would matter first.
- **Answer:** a group estimates the count for a ballot b as poll count − its own previous ballots + its size × b, and judges b by its expected utility gain over its sincere ballots, over the polls. It first tries a few candidate ballots: its previous answer, and the naive exaggerations of its ballot type around the two leaders (A = the one it prefers, B = the other):

  | Ballot | Naive exaggerations |
  |---|---|
  | one name | vote A |
  | approval | approve everyone liked at least as much as A |
  | score, three grades | top grade for everyone liked at least as much as A, bottom for everyone liked no more than B, stretched in between; or top down to A and bottom below |
  | ranking | A to the top and B to the bottom; A to the top; B to the bottom |
  | ranking with ties | everyone liked at least as much as A tied at the top, with or without B at the bottom; or the ranking moves |
  | truncated ranking | rank only down to A, sincerely or with A first |
  | ranking of at most 3 | A first and B never ranked; or rank only down to A (at most 3) |
  | two-round, plurality first round | mark A (the runoff ranking stays sincere) |
  | two-round, approval first round | approve everyone liked at least as much as A (the runoff ranking stays sincere) |
  | veto | veto B |

  **Aggressive** voters take the best of these if it does not lose in the polls: exaggeration wins ties, they exaggerate when it cannot hurt, whatever the gap. **Kind** voters take it only if it strictly gains: honesty wins ties. `--aggressive` (0.5) is the share of strategic voters who are aggressive; each voter's type is drawn per election from its own random stream (the same for every method), and aggressive and kind voters form separate groups. They are only candidates: one that backfires in the polls is not taken. Then random search (at most `--search` proposals, 10; it stops after 3 in a row without gain) takes a change only if it strictly gains. A group none of whose candidates is safe and whose search finds no gain stays sincere. Without a runner-up (exact polls) there are no exaggerations, only the search.
- **Settled:** the last round gave the same ballots as the round before (a fixed point).
- Condorcet-random-drop, which has no feature sums, is polled by tallying the sampled ballots, with the group's sampled members changed.

### Convergent learning (`--converge N`)

The earlier model: over N rounds, groups take turns and each sees the others' latest strategic ballots through a fresh poll of the full electorate. It finds where an electorate ends up after long adjustment, with much more information and coordination than one election offers. That makes it resistant to strategies that only fail when many voters use them at once: mass burial under Borda, for example, which sinks Borda when everyone buries simultaneously (FINDINGS.md). Kept for comparison and as a long-run view; the results up to 2026-10-08 used it with N = 20 and 10 poll samples (`--converge 20 --samples 10`).

- **Groups:** strategic voters with the same sincere ranking, split at random into subgroups of at most `--group-size` voters (see Coordination below). A group either votes sincerely (each member their own sincere ballot) or casts one shared ballot.
- **Polls:** each round draws `--samples` bootstrap resamples of the electorate. A proposal is judged by the group's total expected utility averaged over these samples, so a group gains in proportion to how often it would be pivotal, not only when it is decisive in the exact electorate.
- Voters start sincere. A group that has not deviated yet changes the sincere ballot of its average member (mean utilities of the group); a deviating group changes its current shared ballot, and sometimes (15% of proposals) proposes going back to sincere.
- **Acceptance:** a proposal is kept only if it raises the group's expected utility. A return to sincere is also kept when it is not worse: voters prefer honesty when indifferent. Without this, a deviation that once paid off stays forever once it stops mattering, and sincerity keeps falling with the number of rounds instead of reflecting what strategy actually pays.
- **Rounds:** groups in random order each round; no early stop. Each group tries only one random change per round, so a round without any accepted change does not mean learning has finished, and `settled` (no change in the last round) is a weak sign of convergence.

### Common to both

- **Proposals** (one small change to a ballot; `strategy.mutate`):
  - one name: switch to another candidate
  - ranking: swap two candidates
  - approval: toggle one candidate (never empty or full)
  - two-round ballots: move the round-1 mark to another candidate, or toggle one approval (never empty or full); the runoff ranking never changes
  - veto: move the veto to another candidate
  - ranking with ties: move one candidate to any other level (equal levels are ties)
  - truncated ranking: swap two candidates, or rank one more or one fewer
  - score: set one candidate to a different value 0–max. For ranges above 0–5, half of the proposals use the six levels of the 0–5 scale stretched to the range (0, 20, … 100 for 0–100), so the chance of trying an extreme does not shrink as the range grows and learning is not slowed by the range alone
  - three grades: set one candidate to a different grade
- **Feature sums:** every tally except Condorcet-random-drop depends on the ballots only through sums over voters of per-voter features (totals, first and last places, the pairwise matrix, grade counts, first choices among each subset of candidates, ...). Each method lists its features and decides from their sums. In both models a group's change is evaluated by updating the polled sums instead of recounting. In convergent learning all features are integers, so this gives exactly the same ballots as recounting (checked against the earlier code); poll counts are scaled from a sample and need not be integers.
- **Random streams:** every method learns with a fresh copy of the same random stream per election (common random numbers: the same polls for every method as long as they draw alike), and the candidates' order of moves in strategic nomination is likewise the same for every method. So results do not change when methods are added or reordered, methods that agree on every ballot set agree in the results (IRV and ER-IRV on honest ballots), and paired differences are not blurred by different poll draws.

## Electorates

Each election draws a fresh electorate (`scenario.py`):

- cluster centres: `--clusters` points, normal in `--dims` dimensions;
- cluster sizes: random weights from a flat Dirichlet distribution, so some elections have one dominant cluster and others are balanced;
- voters: each assigned to a cluster by these weights, placed normally around its centre with standard deviation `--spread` (per dimension, before the Zipf scaling);
- candidates: drawn from the same distribution as voters (cluster by the same weights, same spread), unless moved by the two candidate parameters below;
- utilities: `u = exp(-distance / width)`, absolute and not normalised: 1 at zero distance, towards 0 far away. A smaller `--width` makes voters care only about nearby candidates.

**Zipf dimensions** (the default, `--zipf 1`), after Clay Smith's Mixture of Zipf Gaussians ([article](https://cdsmithus.medium.com/simulating-elections-with-spatial-voter-models-1ff50892390), [code](https://github.com/cdsmith/spatial-voting), `src/Model.hs`): dimension i (1, 2, 3, ...) has standard deviation 1/i — for cluster centres, voters and candidates alike — so one issue dominates and the others matter less and less, without being absent.

- Scaling follows Smith's code: the standard deviation is 1/i, so the variance is 1/i². (The article's wording "proportional to 1/n" could also be read as variance 1/n; the code is unambiguous.) The total variance then converges (to π²/6), so beyond about 4 dimensions their number hardly matters. The default of 10 dimensions is well past that.
- Each cluster applies the 1/i scaling to its own slightly shuffled order of dimensions (each rank perturbed by normal noise with standard deviation 0.1 × dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues they vary on most. Cluster centres keep the global order.
- Taken over from Smith: the 1/i scaling and the per-cluster shuffle. Not taken over: his other settings (10 clusters with uniform random weights, cluster centres uniform in ±3/i, a random spread factor per cluster, utility = −distance); our Dirichlet weights, normal centres, one spread and `exp(-distance / width)` stay.
- `--zipf 0` gives the alternative model: all dimensions with standard deviation 1 (then use few dimensions, e.g. 2). Distances are longer there, so the same width gives steeper preferences.

**Candidates unlike the electorate.** Two parameters move the candidates relative to the voters; both act after all random draws, so the voters (and the candidates' own draws) are the same for every value and a sweep compares the same electorates with different fields.

- `--cand-dispersion s` (default 1): each candidate's distance from the voters' mean position is multiplied by s. Below 1 the candidates cluster toward the centre of the electorate; above 1 they are more extreme than the voters. This is the relative candidate dispersion of Merrill (1984), "A comparison of efficiency of multicandidate electoral systems", *American Journal of Political Science* 28, whose Condorcet efficiency of plurality and IRV falls as candidates cluster (centre squeeze: the central candidate is squeezed out of the first choices).
- `--cand-shift d` (default 0): all candidates move along the first, most important dimension by d standard deviations of the voters on it: a systematic bias, e.g. candidates more often from one social group.
- Both change how far the best candidate is from the random ballot, the zero of VSE and WME. With a strong shift, one candidate is nearly everyone's favourite, the random ballot is almost as good as the best candidate, and the efficiencies' denominators become small; compare methods within a condition, and use CE or raw rates across conditions.

## Parameters

`run.py` (one condition). The defaults are the base condition of the latest sweeps (see FINDINGS.md):

| Flag | Default | Meaning |
|---|---|---|
| `--elections` | 200 | electorates |
| `--voters` | 99 | voters per election (odd, so pairwise majorities rarely tie) |
| `--cands` | 6 | candidates |
| `--dims` | 10 | issue-space dimensions |
| `--zipf` | 1 | 1: dimension i has scale 1/i (Zipf dimensions, above); 0: all dimensions equal |
| `--clusters` | 5 | voter clusters |
| `--spread` | 0.5 | standard deviation of voters around their cluster centre (centres have 1), before the Zipf scaling |
| `--width` | 1.0 | distance scale of the utility fall-off |
| `--cand-dispersion` | 1.0 | candidates' distance from the electorate's centre × this (< 1 clustered, > 1 spread out); see Electorates |
| `--cand-shift` | 0.0 | candidates moved along the main dimension by this many voter SDs; see Electorates |
| `--seed` | 1 | random seed; election e uses the stream (seed, e), so all methods face the same electorates |
| `--rounds` | 3 | poll rounds before the vote (0 = everyone sincere); see Strategy engine. Three rounds take voters past the first answer to the honest poll, which leans on the modelling of sincere ballots, toward the state they settle in |
| `--samples` | 10 | polls per round (they stand for the uncertainty); with `--converge`, bootstrap resamples per round |
| `--poll-error` | 2 | voters treat polls as this many times less precise than their sampling error |
| `--aggressive` | 0.5 | share of strategic voters who exaggerate whenever it cannot hurt; the others (kind) only when the polls show a gain (poll rounds only) |
| `--poll-size` | 0 | voters per poll; 0 = half the electorate, rounded to an odd number |
| `--search` | 10 | random ballot changes a group tries per poll round, after the naive exaggerations |
| `--converge` | 0 | N > 0: convergent learning with N rounds instead of poll rounds (e.g. `--converge 20 --samples 10`) |
| `--group-size` | 10 | convergent learning only: most voters that coordinate as one group: a number of voters (1 = every voter alone), a share of the electorate between 0 and 1 (0.05 = 5% of the voters, rounded, at least 1), or 0 = whole ranking group |
| `--strategic` | 0.5 | share of voters who vote strategically (see Partial strategy) |
| `--max-score` | 5 | score ballots range over 0–max (see Score range) |
| `--exits` | 0 | 1: candidates may withdraw and re-enter strategically (see Strategic nomination) |
| `--clones` | 0 | most clones in the race at once, fielded strategically; 0 = none (see Strategic nomination) |
| `--start` | honest | what the first poll shows: `honest` ballots, or `previous`: the result of a previous election (see Starting information) |
| `--participation` | off | also check participation (no-show paradox) on sincere and learned ballots (see Participation); slower |
| `--workers` | all cores | parallel processes |
| `--methods` | all | run only these methods (start from none); method or group names (see Methods), case ignored |
| `--exclude` | none | run all methods except these |
| `--out` | `results` | output prefix: `output/<out>-elections.csv` and `output/<out>-summary.csv` |

`sweep.py` takes the same flags, but every condition flag (`--voters` … `--width`, `--zipf`, `--rounds`, `--samples`, `--poll-size`, `--poll-error`, `--aggressive`, `--search`, `--converge`, `--group-size`, `--strategic`, `--max-score`, `--exits`, `--clones`) accepts several values, and `--seeds` several seeds; it runs the full grid and writes `output/<out>-elections.csv` (default `--out sweep`), one run after another. With `--one-at-a-time`, the first value of every flag is the base, and each other value is run with everything else at the base, instead of the full grid. Conditions that differ only in `--rounds` come from one run: the poll rounds up to the highest are computed once and each requested round is written as its own condition (round r of a longer run is exactly what `--rounds r` gives), so `--rounds 1 2 3 4 5` costs as much as `--rounds 5`.

`analyze.py`:

| Flag | Default | Meaning |
|---|---|---|
| (positional) | | one or more elections CSVs |
| `--out` | `analysis` | output directory (`output/analysis`) |
| `--boot` | 1000 | bootstrap replicates |
| `--compare` | none | pairs `"A:B"` for paired differences |
| `--hide` | `references` | methods or groups left out of the scatter and chain plots; `--hide` alone shows all |
| `--show` | all | only these methods or groups in the scatter and sweep plots (overrides `--hide`), e.g. `--show condorcet STAR` |
| `--errorbars` | off | arrow plots: 95% intervals of each method relative to the average of the methods shown (per bootstrap replicate, the value minus the mean over the shown methods), which removes the variation all methods share; clearly separated bars suggest a real difference, an approximation of the paired comparison. Track plots: each point's own interval |
| `--track` | none | methods or groups to follow across all conditions: `track-<method>.png` each (see below) |
| `--seed` | 0 | bootstrap random seed |

## Robustness: one parameter at a time

To see how much a model assumption matters, vary each parameter on its own around a base condition and check whether the methods keep their order:

```
python sweep.py --one-at-a-time --clusters 5 3 10 --spread 0.5 0.25 1.0 --width 1.0 1.5 \
    --rounds 1 3 --aggressive 0.5 0 1 --strategic 0.5 1 --seeds 1 2 --elections 400 --out model
python analyze.py model-elections.csv --out model-analysis
```

Conditions are labelled by how they differ from the base (`c1: base`, `clusters 5`, ...).

`analyze.py` adds a **rank stability** table when there are several conditions: for each axis, the rank correlation (Spearman) between the methods' values in a condition and in the base (the first condition). The methods hidden in the plots are left out. Its first row is a noise reference: the agreement expected from sampling noise alone between two runs of the base condition (the base's even and odd elections correlated, then scaled up to full size with the Spearman-Brown formula `2r / (1 + r)`). A condition near or above that row orders methods as the base does; a value clearly below it means the parameter changes which methods do well. With few elections everything is noise, and VSE* in particular needs many elections (several hundred per condition): strategic differences between methods are small. Many methods lie close together on each axis, so rank correlation is a strict test; the heatmap shows where the changes come from.

An **electorates** table (and a line in each condition's section) describes the generated electorates themselves, independent of the methods: how often a Condorcet winner exists, how often it also has the highest total utility, how much VSE electing it loses (ratio of sums, as VSE), and how steep preferences are (u(favourite) ÷ u(least liked), median over voters, then median over elections). The third column explains most VSE differences between Condorcet and rated methods; the last one is a concrete handle for choosing `--width`.

## Coordination: what the strategic axes measure

This section applies to convergent learning (`--converge`). Poll rounds always use whole ranking groups; there, coordination is limited by information (one poll, simultaneous answers) instead of by a group-size cap.

How many voters move together decides which strategies pay, and the methods differ sharply in this.

- **Compromise** (voting for a viable candidate instead of a hopeless favourite) pays even for a single voter. It is the main strategy in TTR (top-two runoff) and IRV, and it tends to carry the Condorcet winner into the final round.
- **Burial** (ranking or rating a rival below a candidate one likes less, to create a cycle or to drop the rival in a runoff) only pays when many voters do it together. A single voter out of 99 is almost never pivotal for it. It is the main strategy against Condorcet methods, STAR and BTR-score, and it targets exactly the Condorcet winner.

With `--group-size 0` (whole ranking groups) the strategic axes measure resistance to **organised factional strategy**: every voter with the same ranking follows one recommendation, the case most favourable to burial. With `--group-size 1` they measure resistance to **individual strategy**, where burial almost never pays. The default of 10 lies in between; results under strategy should always be read together with the group size used.

Groups are formed from voters with the same full sincere ranking. With many candidates there are many possible rankings (720 for 6 candidates), so most ranking groups are small whatever the cap, and organised blocs become rare: with 6 candidates and 99 voters, the group size changes little. Coordination by favourite or by cluster would be a different model.

A second, smaller asymmetry: learning is myopic. A burying group does not anticipate that the Condorcet winner's supporters may bury back, or that the cycle it creates can elect a candidate it likes even less (the chicken dilemma). This probably makes burial look somewhat more attractive than it would be to far-sighted voters.

Smaller groups mean more groups, and run time grows with their number.

## Starting information

`--start previous` asks whether a method lets the electorate leave a strategic state it no longer agrees with. The first poll does not show the honest ballots but the result of a **previous election**: one extra (dummy) electorate under the same settings, with its own random draw, run through the same number of poll rounds under the same method. Its final ballots are taken as they are and its candidates relabelled at random as the current election's candidates (a new random assignment per election). Everything else is unchanged: the electorate, the honest results and all later polls are those of the current election, and voters start from their sincere ballots; only the information in round 1 differs. So a sweep `--start honest previous` compares the same elections, and the difference is what the outdated picture costs after 1, 3, 8, ... rounds: how strongly a method holds on to a strategic attractor rather than following the real electorate.

- The same previous election is used for every election of a run, so elections stay independent of each other (no drift from one to the next).
- It needs the original field, so it cannot be combined with strategic nomination, and it is for poll rounds only.
- Cost: one extra election per method and run.
- The previous electorate's candidates sit elsewhere in the issue space than the current ones, but only their counts carry over, relabelled; the current candidates and voters are untouched (a shifted opinion spectrum, with the same candidates, would be a different test).

## Partial strategy

`--strategic p` makes a random share p of the voters strategic; the others always vote sincerely. Each voter is strategic with probability p, drawn from its own random stream per election, so the same voters are strategic for every method. Groups (and their size cap) form only among the strategic voters. p = 1 gives full strategy; p = 0 gives the sincere results. The default 0.5 is an intermediate level, in the spirit of other simulations that report 0, 50 and 100% strategy. Sincerity and the strategy types are averaged over all voters, including the sincere ones.

Run time falls less than in proportion to p: halving the strategic voters removes fewer than half of the groups, since many groups are small.

## Score range

`--max-score m` sets the range of the score ballot to 0–m for every method that uses it (Score, MJ, STAR, BTR-score, MARS, IC-MARS, MARS-T, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score, MMWV+Score, MMWVxScore, STLR, Soft Minimax, Soft Minimax-40, Soft Minimax-L, Soft Minimax-D, INS, Geometric median). The default 0–5 is an implementation choice, not part of these methods; other simulations (e.g. Quinn's VSE) also run 0–10 and 0–100. With a coarse range, sincere voters must give candidates they like differently the same score, so pairwise preferences are lost: this is the cap on WME seen for the rated and hybrid methods. A wider range reduces it. MCA and 321 keep their three grades, and methods on other ballots are unaffected; the electorates are the same for every value, so a sweep (`sweep.py --max-score 5 10 100`) isolates the effect of the range, and its chain plot draws the reachable frontier.

Rules that refer to the range scale with it: ARA approves ratings above the middle, ARA-Top only the top score, Score+ stretches to the full range, Reciprocal score measures support as a share of the top score. Learning: see the score mutation in Strategy engine.

## Strategic nomination

`--exits 1` and `--clones N` let the candidates of the initial draw change the field before the vote (`nomination.py`; after Green-Armytage 2014, "Strategic voting and nomination", *Social Choice and Welfare* 42, with clones added). Nobody new enters.

- **Moves.** Exit (at least two original candidates stay in), re-entry after an exit, fielding a clone of any original in the race (itself included; at most one clone per original and N in the race), withdrawing a clone it fielded. A clone is a copy at a small random offset (0.01 per dimension), so voters rank it next to its original and split between the two.
- **Who moves and why.** Candidates sit in the issue space and value a winner like a voter at their position (exp(−distance/width); themselves 1). A clone counts exactly as its original. A candidate makes a move only if it strictly improves its expected utility under the method's result with **sincere** ballots on the new field: candidates foresee honest voting, not learned strategy.
- **Dynamics.** Rounds in which the candidates take turns in random order, each making its best improving move. A round without a move ends it (an equilibrium); otherwise it stops after 20 rounds. Each method gets its own field. The voters then vote on that field, honestly and with learned ballots as usual.
- **Baseline.** Results are judged on the **original field**: a clone's chance of winning counts as its original's, and withdrawn candidates stay in the comparison. VSE, WME, the random ballot, the best candidate and the reachable frontier are therefore the same as without nomination, and the change in VSE and WME is what nomination does to the outcome. Withdrawing the best candidate counts as a loss.
- **Report.** With nomination on, each condition gets a table per method: how often the honest result changed (Green-Armytage's measure of vulnerability, here after the dynamics rather than for a single move), the average number of exits and clones at the end, and how often the field settled. A sweep `--exits 0 1` or `--clones 0 2` draws chain plots with the frontier.

Cost: each candidate move is checked with one sincere tally on the changed field, a few hundred tallies per method and election; learning remains the larger part.

## Participation

`--participation` checks the no-show paradox (`participation.py`): could some voters get a better result by staying home? It is a diagnostic; nobody actually abstains, and the strategic results do not change (staying home is not among the strategic voters' options).

- **Who stays home.** Voters with the same sincere ranking and the same ballot form a group. For each group, 1, half (rounded up) or all of its members leave out their ballots, while everyone else votes as before. A single voter is rarely pivotal among 99, so the larger shares matter.
- **Failure.** The resulting winner lottery is better for those who stayed home (expected utility, summed over them) than the one with their ballots. Leaving out a ballot means the whole ballot, in a two-round election both rounds.
- **Both ballot sets.** Honest: everyone votes sincerely (the classical participation criterion, for groups). Strategic: everyone keeps the learned ballots and the group drops its learned ballot; a strategic ballot that backfires shows up here.
- **Report** (when the CSV has the columns): per method, the share of elections with a failure (with a 95% interval), the share of voters in failing groups, and where it fails, the mean of the largest gain per absent voter as a share of that voter's utility range (favourite minus least liked).
- Score, Approval and Borda satisfy participation with sincere ballots, so they fail only with learned ones; IRV, TTR and the Condorcet methods (with 4 or more candidates) can fail with sincere ballots too.

Cost: one tally per group and share (feature sums minus the absent ballots), a few hundred per method and election; with all methods at the defaults the run takes about 5% longer.

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

**Reachable frontier.** VSE and WME pull apart: the candidate with the highest total utility is often not the one with the best worst margin, so no rule can reach 100 on both. `analyze.py` computes what any rule could reach in the simulated elections if it knew every voter's sincere utilities: for each weight λ between 0 and 1 it elects, in every election, the candidate with the highest λ × (its VSE contribution) + (1 − λ) × (its WME contribution). Since both axes are ratios of sums with denominators that do not depend on the rule, this traces the exact boundary of the reachable region (lotteries fill in between the points). The plots draw it as a curve and grey out the region beyond it; the report gives its two ends and its most balanced point (highest VSE + WME).

**Reachable region.** The frontier is only the upper-right edge of what can be reached. The whole region is a convex polygon: every rule elects one candidate (or a lottery) per election, so its (VSE, WME) is a point inside the polygon spanned by all combinations. `analyze.py` computes it the same way, for all directions round the circle instead of the upper-right quarter only: for each direction it elects, in every election, the candidate furthest in that direction, and these points are the polygon's corners. The plots grey out everything outside it. This shows that the trade-off is pinned at its ends too: a rule with WME 100 always elects the candidate with the best worst margin, which gives one VSE (up to ties), not anything lower. So a point like VSE 50 / WME 100 is not merely bad but impossible, and the region narrows towards both ends. The report gives the other axis at each end ("with the best WME (100) VSE …"). Two reference points are marked with small crosses where they fall inside a plot: the random ballot (0 / 0 by definition) and a random candidate (the zero of classical VSE; its WME is the average worst margin of the candidates). The report gives the random candidate's values.

A method's distance to the curve is what it loses to having only ballots, not utilities, and to strategy. Faint dashed lines mark 1, 2, 4 and 8 points below the frontier (labelled −1 … −8 at their lower end): each is the frontier shifted down and left by that many points on both axes, so a method on the −4 line could gain 4 points of VSE *and* 4 of WME before it reached the frontier. A strategic shift parallel to these lines is a trade between the two measures; one that crosses them inward is a loss on both. They are drawn inside the region only: beyond the frontier's ends the region bends away. Without a frontier (chain and track plots over conditions with different electorates) the lines follow the 100 / 100 corner instead. For this, run.py stores every candidate's total utility and worst margin per election (columns U_cands, WM_cands).

**Condorcet efficiency (CE)**, reported alongside but not an axis: among elections with a Condorcet winner by sincere utilities, the probability the method elects it. It ignores elections without a Condorcet winner and does not distinguish narrow from clear ones.

VSE, WME and CE use sincere ballots; VSE*, WME* and CE* (in the CSV: VSE_learned, WME_learned, CE_learned) the strategic ballots ("learned": after the poll rounds, or after convergent learning). In all cases the preferences that count are the sincere ones.

**Output files of `run.py`:** `<out>-elections.csv` has one row per (election, method, ballots = sincere or learned) with the run's parameters as columns; the simulator version (column `version`, from `run.VERSION`; the report names it, and warns when CSVs of different versions are combined); the quantities for the axes: U_method, U_best, U_rb (random ballot), U_rand (random candidate, for the usual VSE), WM (the winner's worst margin, share of voters), WM_best, WM_rb; has_cw and p_cw (probability of electing the Condorcet winner); the electorate properties cw_is_util, U_cw and fav_ratio (the same in every row of an election; see Robustness); sincerity, settled and the strategy types. `<out>-summary.csv` and the printed table have one row per method:

| Column | Meaning |
|---|---|
| VSE, WME | voter satisfaction and worst-margin efficiency, sincere ballots |
| VSE_learned, WME_learned | the same with learned ballots |
| CE, CE_learned | Condorcet efficiency, sincere and learned ballots |
| sincerity | mean sincerity of the learned ballots, over all voters (below) |
| settled | poll rounds: share of elections where the last round gave the same ballots as the round before (a fixed point); convergent learning: no group changed its ballot in the last round. Cycles never settle |

`on_front` in the summary is a quick check on point estimates (VSE_learned, WME_learned, sincerity; no uncertainty). The analysis uses the tie-aware "Dominated by" instead (see Analysis).

**Strategy types** (`strategy.strategy_types`), among voters whose learned ballot reverses at least one real preference:
- *compromise*: the favourite is not at the top, and the best-liked top candidate is liked above the voter's mean (a viable second choice raised);
- *push-over*: the favourite is not at the top, and every top candidate is liked below the mean (a weak candidate raised, to steer eliminations or runoffs);
- *burial*: the favourite stays at the top, but a preference further down is reversed.

The three are exclusive. For score ballots, "burial" includes small reversals among lower-rated candidates, so a high burial share can go with high sincerity.

**Sincerity** (`strategy.sincerity`), per voter: for every pair the ballot orders (a above b) while the voter likes b more, add `u[b] − u[a]`; divide by the sum of `|u[a] − u[b]|` over all pairs; sincerity is 1 minus this share. 1 (100%) means the ballot never claims a preference the voter does not hold (true for every sincere ballot, since rounding and leaving candidates level never reverse an order). 0 means every preference is reversed. Example, utilities 0.5 / 0.3 / 0.15 / 0.05: voting for the second choice in plurality gives 0.87, voting for the last gives 0.47. For the Inverse-* methods the preferences that the honest ballot expresses are the reversed ones, so sincerity and the strategy types are measured against those: there, reversing the ballot back is the insincere move.

## Ballots and sincere voting

| Ballot | Form | Sincere ballot |
|---|---|---|
| one name | one candidate | the favourite |
| ranking | full strict ranking | by utility |
| approval | approve or not | every candidate above the voter's mean utility (at least the favourite) |
| score 0–max | integer score per candidate (`--max-score`, default 5) | stretched to the voter's own utility range, rounded |
| two-round, plurality first round | one candidate marked for round 1, plus a ranking for the separate runoff | the favourite marked; ranking by utility |
| two-round, approval first round | approvals for round 1, plus a ranking for the separate runoff | every candidate above the mean approved; ranking by utility |
| veto | one candidate against | the least liked |
| ranking, ties allowed | levels, equal = tied | strict ranking by utility (ties only arise from learning) |
| truncated ranking | the top k ranked, the rest unranked (tied last) | rank every candidate above the mean |
| ranking of at most 3 | up to 3 ranked, the rest unranked (tied last) | the 3 best-liked (all, with 3 or fewer candidates) |
| three grades | good / OK / bad per candidate | stretched to the voter's own utility range, rounded to 0–2 (as the score ballot, with three levels; not affected by `--max-score`) |

**Two-round elections** (TTR, Approval runoff, Approval cond. runoff) have a real, separate runoff: voters come back and vote between the two finalists, independently of their first vote. With two candidates, voting for the preferred one is always best, so the runoff is decided by everyone's sincere preference between the finalists (turnout changes between the rounds are not modelled). The ballot carries that sincere ranking as a fixed part; strategy changes only the round-1 marks. Methods that read the runoff from the same ballot (Supplementary vote, STAR, the BTR methods) are automated runoffs, where strategic voters can also misrank the finalists.

## Methods

73 methods, listed by group (see Groups below). "0–5" in the Ballot column is the score ballot with the default range; `--max-score` changes it for all of these methods at once.

**condorcet**

| Method | Ballot | Rule |
|---|---|---|
| Ranked pairs | ranking | lock pairwise wins from the largest margin down, skipping any that would close a cycle; the candidate with no locked defeat wins. Equal margins are locked in every order and averaged (at most 240 orders, otherwise 240 fixed random ones) |
| Schulze | ranking | strongest path by margins; a candidate whose path to every rival is at least as strong as the rival's path back wins |
| Schulze-ER | ranking, ties allowed | Schulze on rankings with ties (equal-ranked candidates count as neither above the other) |
| Kemeny-Young | ranking | the ranking agreeing with the most pairwise preferences; its top candidate wins (exact, by dynamic programming over subsets); candidates heading different optimal rankings split |
| Copeland | ranking | most pairwise wins (ties count half) |
| Ranked Robin | ranking, ties allowed | Equal Vote's Condorcet method: most pairwise wins; ties go to the greatest sum of margins against the other tied candidates, then against all candidates; remaining ties (its 3rd and 4th tiebreakers) are split. Sincere ballots are strict; ties are learned |
| Ranked Robin-SR | ranking | Ranked Robin on strict rankings (Ranked Robin itself allows ties) |
| SICT | ranking, ties allowed | Symmetrical ICT: p(X,Y) if (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top); X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins; none or all unbeaten: most first places; otherwise most first places among the unbeaten. Sincere ballots are strict; ties are learned |
| Condorcet-random-drop | ranking | Condorcet winner if there is one; otherwise voters are dropped one at a time in random order until the rest have one. Averaged over 32 fixed random orders (the same on every call), so probabilities come in steps of 1/32 |

**minimax-family**

| Method | Ballot | Rule |
|---|---|---|
| Minimax | ranking | smallest worst pairwise defeat (by margin) |
| Minimax-ER | ranking, ties allowed | Minimax on rankings with ties (equal-ranked candidates count as neither above the other) |
| MMWV | ranking, ties allowed | Minimax by winning votes: the worst defeat is measured by the votes for the winner of that pair. On strict rankings identical to Minimax (margins); differs only once equal rankings are learned |
| MMPO | ranking, ties allowed | minimax pairwise opposition: the smallest largest number of voters ranking some rival above the candidate, whether or not that rival wins the pair. Not a Condorcet method; on strict rankings identical to Minimax |
| MMPOWV | ranking, ties allowed | proposed in this lab: (1 − largest pairwise opposition / n) × (1 − worst defeat in winning votes / n); the highest product wins. Multiplying makes a weakness on either costly and is smooth, without thresholds for strategy to aim at. On strict rankings it orders candidates like Minimax, so it equals Minimax with honest ballots and differs only once equal rankings are learned. Not Condorcet-consistent once ties are used |

**condorcet-hybrid**

| Method | Ballot | Rule |
|---|---|---|
| Black | ranking | Condorcet winner if there is one, otherwise the Borda winner |
| Baldwin | ranking | eliminate lowest Borda score, recounted among remaining |
| Baldwin-ER | ranking, ties allowed | Baldwin on rankings with ties (equal-ranked candidates count as neither above the other) |
| Nanson | ranking | repeatedly eliminate every candidate whose Borda score among the remaining is at or below the average (Baldwin eliminates only the lowest) |
| BTR-Borda | ranking | order by Borda score, bottom-two pairwise runoffs climbing up |
| BTR-IRV | ranking | each round the two with the fewest first choices meet pairwise; the loser is eliminated |
| Benham | ranking | IRV, but a Condorcet winner among the remaining candidates wins at once |
| Smith//Score | 0–5 | highest score total among the Smith set (pairwise from the ratings, equal ratings no preference). With a Condorcet winner by the ballots the same as BTR-score; they differ only in cycles |

**plurality-based**

| Method | Ballot | Rule |
|---|---|---|
| Plurality | one name | most first choices |
| TTR | two-round, plurality first round | a candidate marked by more than half the voters wins; otherwise the top two go to a separate runoff, decided by sincere preferences |
| Supplementary vote | ranking (first two used) | top two by first preferences; a ballot whose first preference is out counts for its second preference if that is a finalist, otherwise it is exhausted |
| IRV | ranking | eliminate fewest first choices among remaining |
| ER-IRV | ranking, ties allowed | IRV where every candidate tied at the top of a ballot (among the remaining) gets a whole vote, as in approval. Sincere ballots are strict, so it equals IRV with honest ballots; ties are learned |
| IRV-3 | ranking of at most 3 | IRV as used in practice with a limited ballot: at most 3 candidates ranked; a ballot whose ranked candidates are all out is exhausted and counts for nobody. Equals IRV with 3 candidates |
| DSC | ranking, ties allowed | Descending Solid Coalitions (Woodall): every set of candidates gets the number of voters who rank all of it strictly above all other candidates; from the strongest set down, each makes the candidates outside it ineligible unless none would remain; the last eligible candidate wins. Equal-strength sets whose order matters are applied in every order with equal weight. Clone-independent; with strict rankings the coalitions are each voter's top-k sets |
| DAC | ranking, ties allowed | Descending Acquiescing Coalitions: as DSC, but a voter counts for every set none of whose outside candidates they rank strictly above one inside (equal ranks count for both sides). Equals DSC on strict rankings; differs once voters tie candidates |
| Coombs | ranking | a candidate with more than half of the first choices among the remaining wins; otherwise the one with the most last places among the remaining is eliminated |

**approval-based**

| Method | Ballot | Rule |
|---|---|---|
| Approval | approval | most approvals; sincere: above the voter's mean |
| Approval runoff | two-round, approval first round | top two by approvals go to a separate runoff, decided by sincere preferences; a runoff tie goes to more approvals |
| Approval cond. runoff | two-round, approval first round | as Approval runoff, but a candidate approved by more than half the voters wins outright |
| MDDA | truncated ranking (ranked = approved) | disqualify every candidate that more than half the voters rank some other candidate above (unless all would be); most approved remaining wins. Sincere: rank everyone above the mean |
| MAMPO | truncated ranking | Majority Approval, Minimum Pairwise Opposition (Kevin Venzke): ranked = approved. If nobody is approved by more than half the voters, the most approved wins; otherwise, among those approved by more than half, the one with the smallest largest pairwise opposition (most voters ranking some rival above it) wins. Devised to show that favourite betrayal, strong defensive strategy and strategy-freeness are compatible |

**rated**

| Method | Ballot | Rule |
|---|---|---|
| Score | 0–5 | highest total; sincere: stretched to the voter's range |
| MJ | 0–5 | majority judgment: highest median grade; ties broken by repeatedly removing the median |
| MCA | three grades | majority choice approval, MCA-A variant: the first grade level (good, then good or OK, then all) at which some candidate is rated that high by more than half the voters decides; one such candidate wins, among several the most approved (good or OK) |
| Reciprocal score | 0–5 | interpretation of [electowiki](https://electowiki.org/wiki/Reciprocal_Score_Voting): a voter's faction is their top-rated candidate(s). R(X→Y) is the mean rating supporters of X give Y, as a share of the top score. A score for c is multiplied by min(1, R(c→F)/R(F→c)), F the voter's faction: support that c's supporters do not return is cut. Scores for the voter's own top and for candidates nobody top-rates stay unchanged; highest total wins |
| INS | 0–5 | iterated normalized score, a score elimination: each round every ballot is stretched to the full range over the remaining candidates (lowest 0, highest the top score; a ballot rating them all the same counts for none), the lowest total is eliminated (ties for last split the win), until one is left |
| Geometric median | 0–5 | the geometric median of the score ballots, taken as vectors over the candidates (the point with the smallest total Euclidean distance to all ballots; at most 20 Weiszfeld iterations from the mean); the candidate with the highest coordinate wins. The mean would be Score, the coordinate-wise median a median grade. Each ballot pulls with the same force however far away it is, so exaggerating in the same direction adds no pull. No feature sums, so polls use smaller samples, as for Condorcet-random-drop |

**hybrid**

| Method | Ballot | Rule |
|---|---|---|
| STAR | 0–5 | top two by total, majority runoff; a runoff tie goes to the higher total |
| BTR-score | 0–5 | order by total score, bottom-two pairwise runoffs climbing up |
| MARS | 0–5 | as BTR-score; each match adds share preferring to score as share of maximum |
| IC-MARS | 0–5 | MARS where a pairwise margin counts only beyond the votes rating both candidates at the top (tied at the top, as in [Improved Condorcet Approval](https://electowiki.org/wiki/Improved_Condorcet_Approval)): a margin those votes could reverse counts 0, so score weighs more |
| MARS-T | 0–5 | MARS with each pairwise margin scaled by the share of ballots that score the two candidates differently (1 − share of equal scores): pairs that many voters do not tell apart are decided more by the score difference |
| ARA | 0–5 | ratings above the middle of the range are approvals (0–5: 3–5; 0–10: 6–10); order by approvals; each match adds voters preferring to approvals (formerly MARA) |
| ARA-Top | 0–5 | as ARA, but only the top score counts as approval |
| Score+ | 0–5 | simplified variant from the [votingtheory.org thread](https://www.votingtheory.org/forum/topic/138/score-interval-score-with-additional-protection-against-the-chicken-dilemma): A defeats B robustly if A's total minus the scores both share (per voter the lower of the two) still exceeds B's total. Robustly defeated candidates are eliminated, each voter's scores among the rest are stretched again to the full range, highest total wins |
| 321 | three grades | 3-2-1 voting: 3 semifinalists with the most "good" (the third only with at least half as many as the first), of them 2 finalists with the fewest "bad", then the finalist rated above the other on more ballots. Ties: higher score (good 2, OK 1), then split. The party rule for semifinalists does not apply |
| MMWV+Score | 0–5 | score share (total ÷ (voters × top score)) minus the worst defeat in winning votes as a share of voters; the highest wins (proposed in this lab; pairwise from the ratings) |
| MMWVxScore | 0–5 | score share × (1 − worst winning-votes defeat share), as in MMPOWV |
| STLR | 0–5 | score then levelled runoff: the two highest score totals go on; in the runoff each ballot's scores for the two are rescaled so that the preferred one gets the top score (the other keeps its ratio, e.g. 2 and 1 become 5 and 2.5); a ballot giving both 0 counts for neither; a runoff tie goes to the higher total |
| Soft Minimax | 0–5 | proposed in this lab: Minimax on soft margins. For each pair, every ballot's score difference is clipped to ±20% of the range and divided by that, so a difference of a fifth of the range or more is a full vote for one side and smaller ones count in proportion; the candidate whose worst soft margin is best wins. With the default 0–5 scale the clip is 1 point, so on integer scores this is exactly Minimax by margins on the score ballots; with a wider range (`--max-score 10`) small differences count in part. Clipping at the whole range would give Score |
| Soft Minimax-40 | 0–5 | the same with the clip at 40% of the range (2 points on 0–5): 1-point differences count half; between Soft Minimax and Score |
| Soft Minimax-L | 0–5 | Soft Minimax with an adaptive clip per pair: the mean of the two candidates' average scores (A averages 3, B 2: clip 2.5 points). Pairs of well-liked candidates are compared more by score difference, pairs of little-liked ones more by majority vote |
| Soft Minimax-D | 0–5 | Soft Minimax with an adaptive clip per pair: 1 point plus the difference of the two candidates' average scores. Pairs close in average score are compared by majority vote, pairs far apart more by score difference |

**positional**

| Method | Ballot | Rule |
|---|---|---|
| Borda | ranking | points by position |
| Borda-ER | ranking, ties allowed | a point for every candidate ranked strictly below; ties score nothing. Equals Borda on strict rankings |
| Dowdall | ranking | positional, 1, 1/2, 1/3, … points by rank (Nauru) |

**references**

| Method | Ballot | Rule |
|---|---|---|
| Random pair runoff | ranking | two random ballots; same first choice wins, else majority runoff between the two |
| Anti-plurality | veto (one name against) | fewest vetoes wins; sincere: against the least liked |
| TRASH | ranking | "This Random Acronym Sounds Honest": the plurality runner-up and the anti-plurality runner-up (second fewest last places) meet in a runoff, and the one ranked lower by more voters wins; if both are the same candidate, it wins. A deliberately bad reference |
| COWPEA Lottery | approve or not | single-winner COWPEA lottery: a random ballot's approved candidates; further random ballots each remove the candidates they do not approve (a ballot that would remove all, or none, is skipped) until one remains; if no ballot can break a tie, the remaining share the win. Exact probabilities, not a simulation. Made for proportional representation; here a lottery reference next to Random pair runoff |

**inverse**: tests of the strategy model. Each has the ballot and the count of its original, but its sincere ballot expresses reversed preferences, so honest voting elects (nearly) the worst candidate. Voters who learn to reverse their ballot get the original method back; how far the poll rounds get there shows how well the strategy model finds strategies that are not small steps from the honest ballot.

| Method | Ballot | Rule |
|---|---|---|
| Inverse-Plurality | one name | most named wins; the sincere ballot names the least liked candidate |
| Inverse-Approval | approve or not | most approvals win; the sincere ballot approves every candidate below the voter's mean utility |
| Inverse-Minimax | ranking | Minimax; the sincere ballot ranks upside down, so honestly the Condorcet loser (if any) wins |
| Inverse-IRV | ranking | IRV; the sincere ballot ranks upside down |
| Inverse-ER-IRV | ranking, ties allowed | ER-IRV; the sincere ballot ranks upside down (strictly, like every sincere ballot); strategic voters can tie candidates |
| Inverse-TTR | two-round, plurality first round | round 1 as TTR, the sincere ballot marks the least liked. The runoff is a separate two-candidate vote in which voters vote by their real preference whatever the ballot means: the fixed runoff part of the ballot is the reversed ranking and the count reads it back, so the real majority decides. Only round 1 can be learned, as in TTR |
| Inverse-Borda | ranking | Borda; the sincere ballot ranks upside down |
| Inverse-Score | 0–5 | Score; the sincere ballot scores upside down (the least liked gets the top score) |

**Groups.** Every method belongs to exactly one group. Groups can be used wherever methods are named (`--methods`, `--exclude` in run.py and sweep.py; `--show`, `--hide`, `--track` in analyze.py), they are the panels of the chain plots, and the report and summary.csv show each method's group.

| Group | Methods |
|---|---|
| `condorcet` | Ranked pairs, Schulze, Schulze-ER, Kemeny-Young, Copeland, Ranked Robin, Ranked Robin-SR, SICT, Condorcet-random-drop (Condorcet methods decided by the pairwise counts alone) |
| `minimax-family` | Minimax, Minimax-ER, MMWV, MMPO, MMPOWV (the minimax family; MMPO and MMPOWV are not Condorcet methods) |
| `condorcet-hybrid` | Black, Baldwin, Baldwin-ER, Nanson, BTR-Borda, BTR-IRV, Benham, Smith//Score (Condorcet methods built on a Borda, IRV or score count) |
| `plurality-based` | Plurality, TTR, Supplementary vote, IRV, ER-IRV, IRV-3, Coombs, DSC, DAC (first choices, eliminations and coalitions of top preferences; Coombs eliminates by last places) |
| `approval-based` | Approval, Approval runoff, Approval cond. runoff, MDDA, MAMPO |
| `rated` | Score, MJ, MCA, Reciprocal score, INS, Geometric median |
| `hybrid` | STAR, BTR-score, MARS, IC-MARS, MARS-T, ARA, ARA-Top, Score+, 321, MMWV+Score, MMWVxScore, STLR, Soft Minimax, Soft Minimax-40, Soft Minimax-L, Soft Minimax-D (ratings combined with pairwise comparisons or runoffs) |
| `positional` | Borda, Borda-ER, Dowdall |
| `references` | Random pair runoff, Anti-plurality, TRASH, COWPEA Lottery (benchmarks, not proposals; hidden in the plots by default) |
| `inverse` | Inverse-Plurality, Inverse-Approval, Inverse-Minimax, Inverse-IRV, Inverse-ER-IRV, Inverse-TTR, Inverse-Borda, Inverse-Score (tests of the strategy model, below; hidden in the plots by default) |

**Comparison sets**, selection names across groups (not plot panels): `er-pairs`, each strict method next to its equal-ranking version: IRV/ER-IRV, Borda/Borda-ER, Baldwin/Baldwin-ER, Minimax/Minimax-ER and MMWV, Schulze/Schulze-ER, Ranked Robin-SR/Ranked Robin. Equal-ranking versions equal the strict ones with honest ballots (sincere ballots are strict), so these pairs differ only in what strategic voters do with ties.

Example: `--methods condorcet STAR`, `--methods er-pairs` or `--exclude references`.

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.

## Analysis

`analyze.py` reads one or more elections CSVs. Rows with the same parameters except the seed form one **condition**; seeds are pooled as replications.

**Combining runs.** Each method's result in an election depends only on the parameters, the seed and the election number: every method gets the same random streams, whichever other methods run alongside (a test checks this). So results of separate runs can be pooled, and a new or changed method compared with an existing run, such as the example output, without rerunning the other methods: run it with the same parameters and seed, and give both files to `analyze.py`, the newer last.

```
python run.py --methods "My method" --out mine          # added to methods.py; same parameters and seed as the reference
python analyze.py results-elections.csv mine-elections.csv --out compare
```

How the files are combined (the report lists what was done, and `analyze.py` prints it):
- **Same elections:** rows of the same election (same parameters, seed and election number) must describe the same electorate. If they do not, the files come from incompatible simulator versions and the analysis stops.
- **Duplicates:** a method that appears in several files keeps its rows from the last file, for the elections both have. The report says whether its results changed, which shows whether the method (or the simulator) was changed between the runs.
- **Coverage:** within a condition, only the elections every method has are used, so all comparisons stay paired. A run with fewer elections than the reference therefore shortens the whole condition to its elections; a run with another seed has no elections in common with the reference and stops the analysis.
- More elections for the same methods come from another seed (`--seed 2`); the seeds are pooled.
- A method taken from a run without `--participation` shows no values in the participation table.

**Uncertainty: paired bootstrap.** All methods face the same electorates, so each bootstrap replicate redraws whole elections with replacement and recomputes every method on that same draw (1000 replicates by default). This keeps comparisons paired: two methods can differ clearly even when the interval of each value on its own is wide, because elections differ much more from each other than methods differ within an election.

**The front: VSE\*, WME\* and sincerity.** What counts is the result with strategic voters and how honest they can afford to be. Honest VSE and WME depend more on modelling assumptions (who votes how sincerely is itself a model) and are reported for context, not for dominance. The report lists the front (methods dominated by none) above each condition's table, which is sorted by VSE\*.

**What is reported per condition and method:**

| Output | Meaning |
|---|---|
| VSE, WME, VSE*, WME*, CE, CE*, sincerity | point estimate and 95% interval of each value on its own |
| dominated by | methods clearly better on at least one of the three front axes (VSE*, WME*, sincerity) and clearly worse on none, each with the axes on which it is better; the report shows at most three, the CSV all. Clearly: the paired 95% interval of the difference excludes 0, smaller differences count as ties. Without this, methods level on one axis would be kept apart by noise-sized differences. Sincerity is measured very precisely, so at equal VSE* and WME* the more honest method dominates |
| paired differences (`--compare "A:B"`) | difference A − B per axis with a 95% interval; marked as a tie when the interval includes 0 |

**How to interpret:**
- The values are the result. Whether a method lies on the Pareto front is already interpretation: it depends on which axes count and on which methods are included. "Dominated by" is only a pointer to where to look.
- The axes are kept separate on purpose. Each carries its own value judgement (utilitarian or majoritarian, honest or strategic); no weighting into a single score.
- Read results **per condition first**. The strategic axes depend on how voters strategise: the number of poll rounds, the aggressive and strategic shares (and, with `--converge`, the group size; see Coordination).
- An ordering between two methods only counts if their paired difference is not a tie.

**Plots** (in `--out`):

| File | Shows |
|---|---|
| `arrows-<c>.png` | VSE/WME plane (everything outside the reachable region greyed out), every method's shift from honest (open) to learned ballots (filled, labelled) in one picture; filled points coloured by sincerity (blue = sincere, through purple to dark red = less honest; the scale is capped at the 10th percentile of the plotted methods); faint dashed lines 1, 2, 4 and 8 points inside the frontier |
| `shift-<c>.png` | the same shifts as one row per method, VSE and WME panels; easier to read per method |
| `sincerity-<c>.png` | VSE* and WME* against sincerity |
| `axes-heatmap.png` | VSE\*, WME\*, sincerity, VSE and WME, methods × conditions (when there are several; methods present in every condition, sorted by VSE\*) |
| `track-<method>.png` | with `--track`: one method across all conditions in the VSE*/WME* plane (learned ballots filled, honest open). One varied parameter: a chain in its order; two (a grid, e.g. `--aggressive` × `--rounds`): lines along the parameter with more values, colour for the other; more: points with condition tags |
| `chain-<parameter>.png` | sweeps: for each parameter varied around the base (the first condition), each method's learned values as a chain of arrows through the parameter's values, light to dark; one panel per method group, all on the same scale. If the parameter only affects voters' strategy (strategic and aggressive shares, poll rounds, polls, group size), the electorates are the same and each chain starts at the method's honest point; otherwise the honest values form their own dashed chain |

Sweep plots appear automatically for every parameter that has at least two conditions differing from the base only in that parameter: a one-at-a-time sweep (`sweep.py --one-at-a-time`) or a grid over a single parameter. They show point estimates only; with few elections the zigzags in a chain are mostly noise. One far-off method (often Plurality) sets the shared scale for all panels; `--hide` or `--show` zooms in.

Methods are identified by labels, not colour. Random pair runoff, Anti-plurality and TRASH are left out of the scatter plots by default (`--hide`), since their distance compresses the scale; they stay in tables and heatmaps. `--hide` with no names shows all.

Sample size: with 160 elections per condition the intervals of single values are about ±3 points and many paired differences are still ties. For conclusions, several hundred elections per condition and at least two seeds are advisable.
