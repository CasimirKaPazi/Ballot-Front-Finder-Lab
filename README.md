# Voting Pareto Lab

A simulator for comparing single-winner voting methods: plurality, runoffs, IRV, Condorcet methods, score and approval methods and their hybrids, 65 in all, plus eight test methods. It asks how good the winner of each method is, not only when everyone votes honestly but also when voters respond strategically to polls, and how honest voters can afford to be under each method.

**The question.** Every voting method turns ballots into a winner, and every method can be gamed to some degree. Comparisons with honest ballots alone flatter methods that are easy to manipulate; comparisons under fully rational strategy depend on assumptions nobody believes. This lab places voters in between: some of them read polls and adjust their ballots when it pays, the way people vote tactically in real elections, and it measures what the method makes of that.

**How a simulation runs.**

1. **An electorate.** Voters and candidates are points in an issue space (by default six issues, with voters gathered in clumps within clumps). A voter likes a candidate less the farther away it is. Each election draws a new electorate (README: Electorates).
2. **Honest ballots.** Each voter fills in the method's ballot sincerely: a name, a ranking, approvals or scores, depending on the method (Ballots and sincere voting).
3. **Poll rounds.** Polls show the method's own counts from a sample of the electorate. Strategic voters (by default half the electorate), in groups with the same preferences, change their ballot if that improves the result for them in the polls; all groups answer the same poll at once, and the next poll shows the new ballots (Strategy engine).
4. **Optionally**, candidates may withdraw or field clones beforehand (Strategic nomination), and a diagnostic checks whether some voters would do better by staying home (Participation).

**What is measured.** Two efficiencies, both 0 for a random ballot and 100 for the best candidate of each election (Axes):

- **VSE**, voter satisfaction efficiency: the winner's total utility (how well the electorate as a whole likes it);
- **WME**, worst-margin efficiency: the winner's smallest head-to-head margin against any rival (how strongly a majority could object that another candidate should have won).

The two pull apart, so no method can reach 100 on both; the lab computes the boundary that any rule could reach knowing everyone's true preferences (the reachable frontier). Each value comes with sincere ballots (VSE, WME) and after the poll rounds (VSE*, WME*), together with the **sincerity** of the strategic ballots. The main comparison is on VSE*, WME* and sincerity: how good the result is with strategic voters, and how honest they can be.

**What you get.** `run.py` and `sweep.py` produce CSV files with one row per election, method and ballot set; `analyze.py` turns them into a Markdown report with paired bootstrap intervals, the methods no other method beats (the front), paired comparisons and plots: how strategy moves each method in the VSE/WME plane, how methods change across parameter sweeps, and where they stand relative to the frontier.

The model builds on voter satisfaction efficiency (Jameson Quinn's VSE simulations) and on Chris Smith's spatial voter model; in the strategy model, rounds of best responses to the previous poll (level-k reasoning), the leader rule of Myerson and Weber (1993) emerges instead of being coded. Its assumptions are spelled out in the sections below, and eight Inverse-* methods (honest voting elects the worst candidate) test whether voters learn their way back.

Python 3.9 or newer + NumPy; the analysis also needs matplotlib (`pip install -r requirements.txt`). This file describes the software: the model, its parameters, the methods and the analysis. What the simulations have shown so far, with the commands that produced it, is in [FINDINGS.md](FINDINGS.md).

```
python test_methods.py                                   # hand-built checks
python regression.py                                     # same numbers as before? (after a code change)
python run.py --out base                                 # one condition at the defaults -> output/base/
python sweep.py --rounds 1 2 3 --aggressive 0 0.5 1 --out rounds   # grid of conditions -> output/rounds/
python analyze.py rounds --compare "BTR-score:STAR"      # -> output/rounds/analysis/
```

**One folder per run.** Every run or sweep writes into its own folder, and its analysis goes into an `analysis/` folder inside it, so results and their analysis stay together and can be copied or published as one unit:

```
output/base/                     run.py --out base
    elections.csv      one row per election, method and ballot set (the data)
    summary.csv        one row per method
    table.txt          the printed table
    command.txt        the exact command, simulator version and start time
    analysis/          written by python analyze.py base: report.md, summary.csv, plots, command.txt
output/lazy/                     sweep.py --lazy 0 0.5 1 --seeds 1 2 --out lazy
    runs/01-lazy=0.0-seed=1.csv  one elections CSV per run: its number and the varied parameters
    runs/02-lazy=0.0-seed=2.csv  (poll rounds of one run share its file)
    ...
    command.txt
    analysis/          written by python analyze.py lazy (all files in runs/)
```

`--out` names the folder; a name without a directory goes into `output/` (created when needed), one with a directory (`--out runs/test`) is used as given. Without `--out` the folder is named by the time (`run-20261011-143000`, `sweep-…`), so nothing is overwritten by accident; an existing run is only replaced with `--overwrite`. `analyze.py` takes run or sweep folders (a sweep folder means all its run files) or elections CSV files, and looks up bare names in `output/`. Since a sweep keeps one file per run, single runs can be analysed alone, moved, or mixed with runs of other sweeps (see Combining runs). With several inputs (combining runs) it needs `--out` for the analysis folder.

**Progress, saving and resuming.** While a run or sweep works, a line shows the elections done, the time so far and an estimate of the time left (redrawn in place after every election in a terminal; when the output goes to a file, a new line at every tenth of the way). Each election is written to `elections.csv` as soon as it is done, so nothing finished is lost, and a run in progress can already be analysed. If a run stops (Ctrl-C, a crash, the computer shut down), the same command with `--resume` (and the same `--out`) continues it: complete elections are kept, an incomplete last one is dropped, and only the missing ones are run. Since every election depends only on its parameters, seed and number, the result is identical to an uninterrupted run, row for row. After Ctrl-C the exact command to continue is printed. `--resume` stops if the command differs (other methods, parameters or simulator version).

200 elections with all methods at the default settings (three poll rounds, 10 polls per round) took about 20 minutes in one measurement; time scales linearly with cores, with the number of poll rounds and with the polls per round. To try a new method, there is no need to rerun the others: see Combining runs. More candidates and more voters take longer. Use `--methods` or `--exclude` to run fewer, by name or by group (see Methods).

## Files

| File | Contents |
|---|---|
| `models.py` | Electorate models (one entry per model in `MODELS`): positions of voters and candidates; distances and absolute utilities `exp(-distance / width)`, common to all models |
| `show_electorates.py` | Plots and numbers of the electorates a model draws, without methods: example elections, dimensionality, steepness, utilities, Condorcet winner (see Electorates, Looking at the electorates) |
| `scenario.py` | The settings of a condition's electorates (model and parameters) |
| `ballots.py` | Ballot types, one entry per type in `BALLOTS`: encoding, sincere ballots, laziness, strategic changes and exaggerations, implied values and the worst ballot (for fidelity and sincerity) |
| `methods.py` | Exact tallies on ballot matrices (ties shared), method groups, renamed methods, add-on loader |
| `addons/` | Methods in separate files, loaded at start (see Methods, Add-ons); `_template.py` to start from |
| `ADDONS.md` | Tutorial: writing your own method as an add-on |
| `strategy.py` | Learning engine, sincerity and strategy types |
| `nomination.py` | Strategic nomination: exits, re-entry and clones from the initial field |
| `participation.py` | Participation check (no-show paradox) |
| `outcomes.py` | What an outcome is (winner probabilities) and what it is worth, `value(P, v)`: the one place all judging of outcomes goes through |
| `metrics.py` | Efficiencies for three baselines, winner's worst margin, Condorcet efficiency, electorate properties, Pareto front |
| `run.py` | Tallies every electorate with sincere and learned ballots, runs in parallel, writes CSV with the run's parameters as columns |
| `sweep.py` | Runs a grid of conditions (or one parameter at a time) and seeds, one CSV per run |
| `analyze.py` | Paired bootstrap, intervals, dominance pointers, paired comparisons, rank stability, Markdown report |
| `plots.py` | The analysis plots |
| `test_methods.py` | Hand-built checks |
| `regression.py`, `regression.csv` | Regression check: a few tiny fixed runs compared with the stored reference (see Checking changes) |
| `FINDINGS.md` | Results so far and their interpretation |

## Checking changes

Two checks, both quick:

- `python test_methods.py`: hand-built checks. Each method on small profiles with a known winner, the ballot types, truncation and laziness, sincerity and fidelity, the add-ons.
- `python regression.py`: does the simulator still give the same numbers? It runs six tiny conditions (two elections each: the defaults, five candidates, laziness with scores 0..10, convergent learning, strategic nomination, the first poll from a previous election) with every method, including the references, the Inverse-* methods and the add-ons, and compares every column of every row with `regression.csv`. About a minute or less.

A change that only reorganises the code (moving, renaming, splitting files) must leave `regression.py` at "all equal". A change meant to alter results (a fixed bug, a new rule) shows up as differences, listed by method and column; once they are understood, `python regression.py --update` writes the new reference, and `regression.csv` is committed together with that change, so the history shows which commit changed which numbers. New methods (a new add-on) are listed but not counted as differences; `--update` adds them. Numbers count as equal within a relative tolerance of 1e-9, since the last digits can differ between numpy versions or machines.

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

Each election draws a fresh electorate from the model chosen with `--model` (`models.py`, one entry per model in `MODELS`). A model draws only the positions of voters and candidates (and, if it has them, how much each person weighs each dimension); distances, utilities and the two candidate parameters below are the same for every model. The model and its own parameters are written into every row and named in the report; runs from before models existed are read as `clusters`.

Five models are available: `noise` (the default), `clusters`, `field`, `ks` and `ks-quinn`. Each is described below, the default first.

**Fractal noise** (`--model noise`, the default). Voters are dense where a fractal noise is high, like Perlin noise with several octaves, so they gather in clumps within clumps at every scale. There is no number of clusters, no decay law for the dimensions and no salience to choose:

- The noise, new for every election: five octaves of eight plane waves each, in random directions in the `--dims`-dimensional issue space. From one octave to the next the frequency doubles and the amplitude falls by 2^−0.5. (This is fractal noise built from random waves; Perlin's own lattice construction is impractical in more than three dimensions.)
- People are drawn from a standard Gaussian weighted by exp(`--noise-strength` × noise). At 0 that is a plain Gaussian blob; larger values give sharper clumps, and electing the Condorcet winner becomes costlier while a Condorcet winner stays about as frequent.
- Sampling: 2000 points start from the Gaussian; the strength is raised in ten steps, and each time the points are resampled by their gain in weight and then take five small random steps (Metropolis), so that clumps get their real width. Voters and candidates are distinct points drawn from the result.
- Positions are scaled so the dimensions' total variance is π²/6, as `--fixed-variance` does for clusters. Candidates are drawn like voters. Drawing one electorate takes about 0.1 s, longer than the other models, but little next to the methods' strategy.
- The octaves, waves, amplitude decay and sampling settings are fixed in `models.py` (`NOISE_*`). `--noise-strength` is the knob that matters.

**Defaults, and why** (2026-10-16; FINDINGS.md, Electorate models). The defaults were chosen so that the electorates are plausible against two kinds of real data and still separate the methods:

- `noise`, because it has the fewest assumptions of the models and gives a balanced electorate: `clusters` is clumpy (with few clusters, islands), `ks` is uniform and low-dimensional (two or three dimensions drawn, in which VSE and WME hardly pull apart), and `field` gives a featureless blob.
- Strength 5, 6 dimensions, width 0.7. A Condorcet winner exists in about 97% of elections, as in real ranked elections with 50 to 100 or more voters (96–98% in Myers' CIVS data). The candidates' map from approval ballots (`--sav`, below) is close to Alós-Ferrer & Granić's real one with 6 candidates, if somewhat more concentrated on the first dimension. Electing the Condorcet winner loses about 15% of VSE, so VSE and WME pull apart and the methods differ. From strength 5 to 7 the method order stays within its sampling noise, so the exact value matters little. Below 5 (with 10 dimensions and width 1.0, below 6) the electorates become easy: nearly every method is close to the best possible result.
- The models' own defaults are those of their best variant in the comparisons: `clusters` with 20 clusters, no Zipf, fixed total variance and salience per cluster with exponent 0.5; `field` and `ks` unchanged.
- Before 2026-10-16 the defaults were `clusters`, 10 dimensions with Zipf, 5 clusters, no salience, unscaled positions and width 1.0; to reproduce older results, give these explicitly (each results file records every parameter).

**Voter clusters** (`--model clusters`):

- cluster centres: `--clusters` points, normal in `--dims` dimensions;
- cluster sizes: random weights from a flat Dirichlet distribution, so some elections have one dominant cluster and others are balanced;
- voters: each assigned to a cluster by these weights, placed normally around its centre with standard deviation `--spread` (per dimension, before the Zipf scaling);
- candidates: drawn from the same distribution as voters (cluster by the same weights, same spread), unless moved by the two candidate parameters below;
- utilities: `u = exp(-distance / width)`, absolute and not normalised: 1 at zero distance, towards 0 far away. A smaller `--width` makes voters care only about nearby candidates.

**Zipf dimensions** (`--zipf 1`; the default until 2026-10-16), after Chris Smith's Mixture of Zipf Gaussians ([article](https://cdsmithus.medium.com/simulating-elections-with-spatial-voter-models-1ff50892390), [code](https://github.com/cdsmith/spatial-voting), `src/Model.hs`): dimension i (1, 2, 3, ...) has standard deviation 1/i — for cluster centres, voters and candidates alike — so one issue dominates and the others matter less and less, without being absent.

- Scaling follows Smith's code: the standard deviation is 1/i, so the variance is 1/i². (The article's wording "proportional to 1/n" could also be read as variance 1/n; the code is unambiguous.) The total variance then converges (to π²/6), so beyond about 4 dimensions their number hardly matters.
- Each cluster applies the 1/i scaling to its own slightly shuffled order of dimensions (each rank perturbed by normal noise with standard deviation 0.1 × dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues they vary on most. Cluster centres keep the global order.
- Taken over from Smith: the 1/i scaling and the per-cluster shuffle. Not taken over: his other settings (10 clusters with uniform random weights, cluster centres uniform in ±3/i, a random spread factor per cluster, utility = −distance); our Dirichlet weights, normal centres, one spread and `exp(-distance / width)` stay.
- `--zipf 0` (the default) gives all dimensions standard deviation 1. Distances are longer there, so without `--fixed-variance 1` the same width gives steeper preferences.

**Where people stand, and what they care about** (two options of `clusters`):

- `--fixed-variance 1` scales all positions by one factor so that the dimensions' total variance is π²/6, the limit of Zipf dimensions, whatever their number. Without it and without Zipf, every added dimension makes all distances longer, so preferences get steeper at the same width; with it, more dimensions spread the same disagreement over more issues. (With Zipf the factor is close to 1.)
- `--salience cluster` separates what people care about from where they stand: each cluster weighs the issues 1, 1/2, 1/3, … (Zipf, as Smith's dimensions; `--salience-exponent s` gives 1/i^s) in its own random order, in its members' and its candidates' distances, scaled to a mean squared weight of 1. Society can then disagree on many issues while each group mainly cares about a few, and groups differ in which. The positions are the same with and without it (its random draws come last).
- Meant together with `--zipf 0`: positions spread alike on every issue, salience differs by group. This is the default of `clusters`: `--clusters 20 --zipf 0 --fixed-variance 1 --salience cluster --salience-exponent 0.5`.

**Scale-free fields** (`--model field`, experimental). Instead of imposing clusters or decay laws, opinions arise as fields over a social space, following what is known about how they spread:

- People live on a two-dimensional social space (who meets whom; a 64 × 64 torus), placed by a Lévy flight, so they are clustered at all scales.
- Each issue is a Gaussian random field over that space with power spectrum k⁻²: correlations that fall off with the logarithm of distance, as Borghesi & Bouchaud (2010, "Spatial correlations in vote statistics: a diffusive field model for decision-making") found in real vote statistics, i.e. structure at all scales without a characteristic one.
- A person's opinion on an issue is the field where they live (40% of the variance; Borghesi & Bouchaud find a third to a half shared) plus an own part, passed through tanh(`--field-kappa` × ·): a saturating response, as a spin in the Ising model. Small values give smooth, nearly Gaussian opinions; large values push every issue towards two camps.
- `--dims` issues with a fixed total variance (as `--fixed-variance` for clusters). `--field-salience g` makes what people care about a field too (weights exp(g × another field)), so neighbours share their priorities partly.
- The exponent, the shared share and the Lévy exponent are fixed in `models.py` (`FIELD_EXPONENT`, `FIELD_SHARED`, `FIELD_LEVY`): in first tests they hardly changed the electorates' properties. `--field-kappa` is the knob that matters: from the smooth end (like ks) to the polarised end (like clusters).
- Issues are independent fields, so no ideology (opinions on different issues going together) is built in; how to let it emerge (overlapping topics, Baumann et al. 2021) is open.

**Quinn's hierarchical model** (`--model ks`), from Jameson Quinn's VSE simulations ([vse-sim](https://github.com/electionscience/vse-sim), `src/vse_sim/voter_models.py`, `KSModel`, "kitchen sink"), with the parameters of its published results. In the clusters model a voter belongs to one cluster on all dimensions at once, so clusters behave like consistent ideologies or parties. Here the issues come in bundles, and factions form in each bundle separately, so cleavages cut across each other: a voter can be in the big faction on one bundle and a small one on another, and groups differ in how much they care.

- Issue bundles: the first weighs 1, each further one a random fraction (Beta(1, 3)) of the one before, until the weight falls below `--ks-cut` (0.2); within a bundle, each further dimension weighs a random fraction (Beta(1.5, 3)) of the one before, down to the same cut. So the number of bundles and dimensions varies by election (with the defaults mostly 2–4 dimensions in all), and few issues matter much.
- Factions, in each bundle independently: a Chinese restaurant process over voters and candidates. Each person starts a new faction with probability α / (i + α), i the number of persons before, else joins the faction of a random earlier person (α = `--ks-alpha`, 1.5). A few large factions and a tail of small ones; their number is not fixed.
- Each faction, per bundle: a position (normal, with a random variance c ~ Beta(3, 1.5)) and a caring level (Beta(3, 1.5)). Its members sit around its position, the tighter the more it cares (offset N(0, 1) × √(1 − caring) per dimension), and weigh that bundle by their caring.
- Distance: each person weighs each dimension by its importance × their faction's caring on that bundle, normalised per person (as in vse-sim). Utilities use `exp(-distance / width)` like every model here (vse-sim uses −distance), so the models differ only in the electorate's structure. Candidates are drawn exactly like voters; the two candidate parameters below apply as for any model.
- Two versions. `ks` is the lab's version, which may evolve: a change comes as a new parameter whose default keeps the earlier behaviour (so older runs stay comparable), or under a new model name. `--model ks-quinn` is kept as close to vse-sim's code as the lab allows and is not to be changed: it uses vse-sim's one random offset per person and bundle (see the last point) and has no `--ks-shared`. Both read `--ks-alpha` and `--ks-cut`. What still differs from vse-sim in both is common to every model here: utilities `exp(-distance / width)` instead of −distance, and the candidate parameters.
- `--ks-shared 1` (a test variant of `ks`, not in vse-sim): the factions of the first bundle are used in every bundle, so cleavages no longer cut across each other; positions and caring are still drawn per bundle.
- Written anew from the algorithm; `ks` has one change: vse-sim's code gives a person one random offset for all dimensions of a bundle (it reads the offset by bundle number), which looks unintended; here each dimension has its own. With the exp utilities and this change, results are not directly comparable with Quinn's published numbers. `--dims`, `--clusters`, `--spread` and `--zipf` do not apply to this model.

**Looking at the electorates.** `show_electorates.py` draws the electorates of one or more models without running any method, in a few seconds, to check by eye that they look as intended and to compare models:

```
python show_electorates.py --model clusters ks ks-quinn --out look       # -> output/look/
python show_electorates.py --model ks --ks-cut 0.05 --width 0.5 --out look-ks
```

It takes run.py's electorate parameters (one value each; `--model` may list several) and the same random streams, so election e of a seed is the electorate run.py uses. It writes `examples.png` (a few elections per model: voters coloured by their favourite and the lettered candidates, on the voters' first two principal components; the Condorcet winner ringed), `summary.png` (over 300 elections: variance explained by the first principal components and how many reach 70%, favourite ÷ least liked, the utilities, the distances, and the VSE lost by electing the Condorcet winner where it is not the utility winner) and `summary.txt` (the same as numbers). For the ks models, whose people weigh the dimensions differently, positions are scaled by the voters' average weights first: the space roughly as an average voter sees it.

With `--sav` it also maps the candidates the way Alós-Ferrer & Granić (2015, "Political space representations with approval data", *Electoral Studies* 39) mapped real parties and candidates from approval ballots in German field experiments, after Laslier's spatial approval voting (2006). This gives a dimensionality comparable with theirs (four or more dimensions to explain 70% for real voters):

```
python show_electorates.py --model clusters ks --cands 9 --sav --sav-noise 0.5 --out look-sav
```

- Each electorate is drawn with `--sav-voters` voters (1000), and every voter approves the candidates above their own mean utility. `--sav-noise s` first blurs each utility by a normal error of s × the voter's own spread of utilities, so approval near the cut-off is partly chance, as in real ballots.
- For each pair of candidates, the overlap of the people approving them, relative to chance: r = N · a(c, d) / (a(c) · a(d)). Squared distances are proportional to ln(r_max / r) (the paper's formula with its smallest admissible α; the most overlapping pair is at 0). Classical multidimensional scaling then gives the inertia each dimension explains; only positive eigenvalues count.
- A pair nobody approves together would be infinitely far apart; it is given half a voter instead, and the share of elections where this happens is reported. Without noise this happens in most elections, which makes the map less reliable.
- `sav.png` and `summary.txt` show the cumulative inertia by number of dimensions next to the paper's four data sets (Messel: 9 parties, 8 candidates; Konstanz: 11 parties, 6 candidates). Use a similar number of candidates (`--cands`), since n candidates allow at most n − 1 dimensions.

**Candidates unlike the electorate.** Two parameters move the candidates relative to the voters; both act after all random draws, so the voters (and the candidates' own draws) are the same for every value and a sweep compares the same electorates with different fields.

- `--cand-dispersion s` (default 1): each candidate's distance from the voters' mean position is multiplied by s. Below 1 the candidates cluster toward the centre of the electorate; above 1 they are more extreme than the voters. This is the relative candidate dispersion of Merrill (1984), "A comparison of efficiency of multicandidate electoral systems", *American Journal of Political Science* 28, whose Condorcet efficiency of plurality and IRV falls as candidates cluster (centre squeeze: the central candidate is squeezed out of the first choices).
- `--cand-shift d` (default 0): all candidates move along the first, most important dimension by d standard deviations of the voters on it: a systematic bias, e.g. candidates more often from one social group.
- Both change how far the best candidate is from the random ballot, the zero of VSE and WME. With a strong shift, one candidate is nearly everyone's favourite, the random ballot is almost as good as the best candidate, and the efficiencies' denominators become small; compare methods within a condition, and use CE or raw rates across conditions.

## Parameters

`run.py` (one condition). The electorate defaults are explained in Electorates (Defaults, and why):

| Flag | Default | Meaning |
|---|---|---|
| `--elections` | 200 | electorates |
| `--voters` | 99 | voters per election (odd, so pairwise majorities rarely tie) |
| `--cands` | 6 | candidates |
| `--model` | noise | electorate model: `noise`, `clusters`, `field`, `ks` or `ks-quinn` (see Electorates; `models.py`) |
| `--dims` | 6 | issue-space dimensions (models `noise`, `clusters`, `field`) |
| `--noise-strength` | 5 | model `noise`: how strongly voters gather where the fractal noise is high (0 = a Gaussian blob; larger: sharper clumps) |
| `--zipf` | 0 | model `clusters`: 1 = dimension i has scale 1/i (Zipf dimensions, above); 0 = all dimensions equal |
| `--clusters` | 20 | model `clusters`: voter clusters |
| `--spread` | 0.5 | model `clusters`: standard deviation of voters around their cluster centre (centres have 1), before the Zipf scaling |
| `--salience` | cluster | model `clusters`: `cluster` = each cluster weighs the issues 1, 1/2, 1/3, … in its own random order (see Electorates) |
| `--salience-exponent` | 0.5 | model `clusters` with `--salience cluster`: weights 1/i^s (1 = Zipf, 0 = all alike) |
| `--fixed-variance` | 1 | model `clusters`: 1 = scale positions so the dimensions' total variance is π²/6 whatever their number |
| `--field-kappa` | 1.5 | model `field`: saturation of opinions (small: smooth, nearly Gaussian; large: two camps per issue) |
| `--field-salience` | 0 | model `field`: strength of the salience field (0 = every issue weighed alike) |
| `--ks-alpha` | 1.5 | model `ks`: faction alpha (higher: more, smaller factions); see Electorates |
| `--ks-cut` | 0.2 | model `ks`: importance below which no further issue bundles or dimensions are drawn (lower: more dimensions) |
| `--ks-shared` | 0 | model `ks`: 1 = the same factions in every issue bundle (a test variant without cross-cutting cleavages) |
| `--width` | 0.7 | distance scale of the utility fall-off |
| `--cand-dispersion` | 1.0 | candidates' distance from the electorate's centre × this (< 1 clustered, > 1 spread out); see Electorates |
| `--cand-shift` | 0.0 | candidates moved along the main dimension by this many voter SDs; see Electorates |
| `--seed` | 1 | random seed; election e uses the stream (seed, e), so all methods face the same electorates |
| `--rounds` | 3 | poll rounds before the vote (0 = everyone sincere); see Strategy engine. Three rounds take voters past the first answer to the honest poll, which leans on the modelling of sincere ballots, toward the state they settle in |
| `--samples` | 10 | polls per round (they stand for the uncertainty); with `--converge`, bootstrap resamples per round |
| `--poll-error` | 2 | voters treat polls as this many times less precise than their sampling error |
| `--aggressive` | 0.5 | share of strategic voters who exaggerate whenever it cannot hurt; the others (kind) only when the polls show a gain (poll rounds only) |
| `--lazy` | 0 | laziness of honest rankings, 0–1 (see Truncation and laziness) |
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
| `--methods` | all but references and inverse | run only these methods (start from none); method or group names (see Methods), case ignored; `all` for every method |
| `--exclude` | none | leave these out (from the default, or from `--methods`) |
| `--out` | time-stamped | folder of the run: `output/<out>/` (see One folder per run) |
| `--overwrite` | off | replace an existing run in `--out` |
| `--resume` | off | continue an interrupted run in `--out` with the same command (see Progress, saving and resuming) |

`sweep.py` takes the same flags, but every condition flag (`--voters` … `--width`, `--zipf`, `--rounds`, `--samples`, `--poll-size`, `--poll-error`, `--aggressive`, `--search`, `--converge`, `--group-size`, `--strategic`, `--max-score`, `--exits`, `--clones`) accepts several values, and `--seeds` several seeds; it runs the full grid, one run after another, and writes one elections CSV per run into `output/<out>/runs/`. With `--one-at-a-time`, the first value of every flag is the base, and each other value is run with everything else at the base, instead of the full grid. Conditions that differ only in `--rounds` come from one run: the poll rounds up to the highest are computed once and each requested round is written as its own condition (round r of a longer run is exactly what `--rounds r` gives), so `--rounds 1 2 3 4 5` costs as much as `--rounds 5`.

`analyze.py`:

| Flag | Default | Meaning |
|---|---|---|
| (positional) | | one or more run or sweep folders (or elections CSV files) |
| `--out` | `analysis/` in the input folder | analysis folder; needed with several inputs |
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
python sweep.py --one-at-a-time --noise-strength 5 3 7 --dims 6 10 --width 0.7 1.0 \
    --rounds 1 3 --aggressive 0.5 0 1 --strategic 0.5 1 --seeds 1 2 --elections 400 --out model
python analyze.py model                                   # -> output/model/analysis/
```

Conditions are labelled by how they differ from the base (`c1: base`, `noise strength 3.0`, ...).

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

`--max-score m` sets the range of the score ballot to 0–m for every method that uses it (Score, MJ, STAR, BTR-score, MARS, IC-MARS, MARS-T, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score, MMWV+Score, MMWVxScore, STLR, Soft Minimax, Soft Minimax-40, Soft Minimax-L, Soft Minimax-D, INS, Geometric median, and add-ons on score ballots). The default 0–5 is an implementation choice, not part of these methods; other simulations (e.g. Quinn's VSE) also run 0–10 and 0–100. With a coarse range, sincere voters must give candidates they like differently the same score, so pairwise preferences are lost: this is the cap on WME seen for the rated and hybrid methods. A wider range reduces it. MCA and 321 keep their three grades, and methods on other ballots are unaffected; the electorates are the same for every value, so a sweep (`sweep.py --max-score 5 10 100`) isolates the effect of the range, and its chain plot draws the reachable frontier.

Rules that refer to the range scale with it: ARA approves ratings above the middle, ARA-Top only the top score, Score+ stretches to the full range, Reciprocal score measures support as a share of the top score. Learning: see the score mutation in Strategy engine.

**Global range and fixed range.** `--max-score` is the global range: every method on score ballots follows it, unless the method fixes its own range with `settings`, e.g. `Method("score", d_sum, ("sum",), settings={"max_score": 10})`. While such a method's ballots are made, changed (strategy) and tallied, its own range is in force; afterwards the global one again. A method with a fixed range is named after it, `Name-X` (`Score-10`), and its rows still show the run's global `max_score`, so the name is what tells them apart. Avoid fixing the range for the base version of a method: a base method that follows `--max-score` stays comparable across runs, and a max-score sweep varies it like all the others (a fixed one stays the same at every value of the sweep). A test checks that every base method follows the global range and that fixed ones are named `Name-X`. A fixed method is equivalent to its base version at that global range: `Score-10` in a run at `--max-score 5` gives exactly the rows of `Score` at `--max-score 10`. No built-in method fixes its range at present. The settings a method can fix are listed in `SETTINGS` in methods.py (so far only `max_score`).

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

**Output files of `run.py`:** `elections.csv` has one row per (election, method, ballots = sincere or learned) with the run's parameters as columns; the simulator version (column `version`, from `run.VERSION`; the report names it, and warns when CSVs of different versions are combined); the quantities for the axes: U_method, U_best, U_rb (random ballot), U_rand (random candidate, for the usual VSE), WM (the winner's worst margin, share of voters), WM_best, WM_rb; has_cw and p_cw (probability of electing the Condorcet winner); the electorate properties cw_is_util, U_cw and fav_ratio (the same in every row of an election; see Robustness); sincerity, settled and the strategy types. `<out>-summary.csv` and the printed table have one row per method:

| Column | Meaning |
|---|---|
| VSE, WME | voter satisfaction and worst-margin efficiency, sincere ballots |
| VSE_learned, WME_learned | the same with learned ballots |
| CE, CE_learned | Condorcet efficiency, sincere and learned ballots |
| sincerity | mean sincerity of the learned ballots, over all voters (below; elections.csv also has `fidelity`, for both ballot sets) |
| settled | poll rounds: share of elections where the last round gave the same ballots as the round before (a fixed point); convergent learning: no group changed its ballot in the last round. Cycles never settle |

`on_front` in the summary is a quick check on point estimates (VSE_learned, WME_learned, sincerity; no uncertainty). The analysis uses the tie-aware "Dominated by" instead (see Analysis).

**Strategy types** (`strategy.strategy_types`), among voters whose learned ballot reverses at least one real preference:
- *compromise*: the favourite is not at the top, and the best-liked top candidate is liked above the voter's mean (a viable second choice raised);
- *push-over*: the favourite is not at the top, and every top candidate is liked below the mean (a weak candidate raised, to steer eliminations or runoffs);
- *burial*: the favourite stays at the top, but a preference further down is reversed.

The three are exclusive. For score ballots, "burial" includes small reversals among lower-rated candidates, so a high burial share can go with high sincerity.

**Sincerity and fidelity.** Two separate questions about a learned ballot: does it go against the voter's own preferences, and how much does it tell about them?

**Sincerity** (`strategy.sincerity`) is about voting against one's own preferences because it pays. Per voter, the *reversal mass* R of a ballot is the sum of `u[a] − u[b]` over the pairs the voter prefers a > b but the ballot puts b above a. Sincerity is `1 − R(ballot) / R(worst)`, where the worst ballot is the most dishonest one the method's ballot format allows for this voter: a vote for the last choice in plurality, a veto against the favourite, the reversed ranking (rankings, and scores with enough levels), the most damaging cut in approval (found exactly, over all cuts), the reversed order in three levels for three grades, the three least liked ranked in reverse for IRV-3. So 1 means no reversed preference and 0 means as far against one's own preferences as this format permits, the same in every format. Ties are not reversals and cost nothing: exaggeration (5, 3, 1, 0 → 5, 5, 0, 0), an approval threshold anywhere, unranked candidates. They say less, not something false, and in approval or plurality there is no single honest place for the threshold anyway. This is the usual definition of a sincere ballot (Brams and Fishburn: no candidate is put above one the voter likes more), scaled per format.

**Fidelity** (`strategy.fidelity`) is about how much of the voter's utility differences the ballot conveys. Utilities are stretched to 0–1 per voter (favourite 1, least liked 0) and compared with the values the ballot implies: scores and grades by their value, rankings evenly spaced, approval and plurality marks as 1 and 0, veto 0 for the vetoed candidate. Over all pairs, fidelity is `1 − Σ |Δu − Δv| / Σ 2 |Δu|` (Δu, Δv the differences in utility and implied value): 1 if the ballot shows the differences exactly; a tie loses half of a pair; a mirrored pair loses all of it. A sincere ranking of utilities 100 / 3 / 2 / 1 / 0 loses a lot, since it implies 100 / 75 / 50 / 25 / 0. Fidelity therefore includes what the format cannot express: the report gives it for the learned ballots and for the sincere ones ("(honest)"), so the format's own loss and what strategy adds can be told apart. It is reported for context, not used for the front.

Example, utilities 0.5 / 0.3 / 0.15 / 0.05:

| Ballot | Sincerity | Fidelity |
|---|---|---|
| Plurality: vote for the 1st choice (sincere) | 1 | 0.72 |
| Plurality: vote for the 2nd choice | 0.75 | 0.32 |
| Plurality: vote for the last choice | 0 | 0.05 |
| Ranking, sincere | 1 | 0.93 |
| Ranking: 2nd moved above the 1st | 0.87 | 0.63 |
| Ranking: reversed | 0 | 0 |
| Score 0–5: sincere 5, 3, 1, 0 | 1 | 0.97 |
| Score 0–5: exaggerated 5, 5, 0, 0 | 1 | 0.70 |
| Approval: {1st, 2nd} (sincere) | 1 | 0.70 |
| Approval: {3rd, 4th} | 0 | −0.10 |

The exaggerated score ballot and the approval ballot say the same and get the same values. Two-round ballots (TTR, the approval runoffs): the voter only chooses the round-1 marks (the runoff ranking is the sincere one), so both measures look at the marks, as a plurality or approval ballot. **Undefined for TRASH and the Inverse-\* methods.** Both measures assume that the honest ballot is the one a voter would want to keep, so moving away from it is a cost of strategy. In TRASH (the candidate ranked lower wins) and the Inverse-\* methods (honest voting elects the worst candidate) the honest ballot works against the voter's own interest: what a ballot says and whom it helps point opposite ways, and leaving the honest ballot is escaping a bad method, not giving up honesty. Sincerity and fidelity are therefore left blank for these methods (`methods.sincerity_defined`), in the CSV, the tables (—) and the plots; on the front they count as ties on the sincerity axis. The strategy types (compromise, burial, push-over) are still reported; for the Inverse-\* methods they are measured against the reversed preferences the honest ballot expresses. CSVs from before 2026-10-12 (no fidelity column) hold an earlier sincerity measure (reversals over all pairs, not scaled per format); the report says so.

**Condorcet winner in the cast ballots** (`metrics.ballot_condorcet`). Studies of real ranked-ballot elections compute the Condorcet winner from the cast ballots, which may themselves be strategic. The model knows both: the real Condorcet winner (sincere preferences) and the one the ballots show (a voter counts for a over b if the ballot puts a higher; equal = no preference). Per election and method (columns `bcw`, `p_bcw`): the ballots show the same Condorcet winner, another one (false), none although there is a real one (a cycle or ties), or one although there is none (spurious); and how often the method elects the ballots' Condorcet winner, which is the Condorcet efficiency ballot data would report. Only for ballots that record an order (rankings, scores, grades); plurality, approval, veto and two-round marks are left blank, since their pairwise counts mostly reflect the marks. For the Inverse-\* methods the ballots are read reversed. The report has a table per condition, with the honest ballots in brackets: for rankings these always show the real winner, for scores and grades rounding and ties can already hide or change it.

## Ballots and sincere voting

| Ballot | Form | Sincere ballot |
|---|---|---|
| one name | one candidate | the favourite |
| ranking | ranking; may be truncated (the top k ranked, the rest unranked, tied last) | by utility, complete (truncated with `--lazy`) |
| approval | approve or not | every candidate above the voter's mean utility (at least the favourite; with `--lazy` only those the voter cares about) |
| score 0–max | integer score per candidate (`--max-score`, default 5) | stretched to the voter's own utility range, rounded (with `--lazy` the others blank = 0) |
| two-round, plurality first round | one candidate marked for round 1, plus a ranking for the separate runoff | the favourite marked; ranking by utility |
| two-round, approval first round | approvals for round 1, plus a ranking for the separate runoff | every candidate above the mean approved; ranking by utility |
| veto | one candidate against | the least liked |
| ranking, ties allowed | levels, equal = tied; level 0 shared by two or more = unranked (truncation) | strict ranking by utility (ties only arise from learning; truncated with `--lazy`) |
| truncated ranking | the top k ranked, the rest unranked (tied last) | rank every candidate above the mean (with `--lazy`: only those the voter also cares about) |
| ranking of at most 3 | up to 3 ranked, the rest unranked (tied last) | the 3 best-liked (all, with 3 or fewer candidates; with `--lazy`: only those the voter also cares about) |
| full ranking (Inverse-\* methods) | full strict ranking, never truncated | by the reversed preferences |
| three grades | good / OK / bad per candidate | stretched to the voter's own utility range, rounded to 0–2 (as the score ballot, with three levels; not affected by `--max-score`) |

**Truncation and laziness.** Rankings may be truncated: the voter ranks the top k candidates and leaves the rest unranked, tied last (a ranking of all but one is the full ranking). Strategic voters can truncate (rank only down to the leader they prefer, or bullet-vote for it) and the random search can rank one candidate more or fewer. Truncation reverses no preference, so it does not lower sincerity; it lowers fidelity (less is said). How the methods treat unranked candidates:

| Methods | Unranked candidates |
|---|---|
| pairwise (Schulze, Ranked pairs, Minimax family, Kemeny-Young, Copeland, Ranked Robin, Baldwin, Nanson, SICT, Condorcet-random-drop, …) | below every ranked candidate, tied with each other (no preference between them) |
| IRV, ER-IRV, Benham, BTR-IRV, Coombs (first choices) | a ballot with no ranked candidate left is exhausted and counts for nobody; majorities are of the continuing ballots |
| Coombs (last places) | the unranked share the last place equally |
| Borda, Black, BTR-Borda | 0 points; ranked candidates keep their full points (C−1, C−2, …), so truncating can pay |
| Dowdall | 0 points |
| Supplementary vote | a ballot with only one candidate ranked has no second preference |
| TRASH (last places) | shared, as in Coombs |

In the ER variants (rankings with ties) a tie at level 0 of two or more candidates means unranked; ties higher up keep the equal-ranking rules (in ER-IRV each candidate tied at the top among the remaining gets a whole vote). The Inverse-\* methods keep full strict rankings: their honest ballot is the reversed one, and truncating it has no clear meaning.

`--lazy λ` (0–1) makes honest rankings truncated: a voter ranks a candidate only if its utility, stretched to 0–1 between the voter's least and most liked candidate, is at least λ (the favourite always). λ = 0 gives full rankings, λ = 1 bullet votes; in between the length depends on the voter's preferences (a voter with one clear favourite ranks fewer than one who likes several about equally), so it is neither a share of voters nor a fixed number of ranks. It applies to every ballot that can say more than one thing: rankings (ranking, ranking with ties, truncated ranking, ranking of at most 3; for the last two on top of their own cut) are truncated; approval ballots (also the round-1 approvals of the approval runoffs) approve only the candidates above the mean that the voter cares about, so λ = 1 is a bullet vote; score and grade ballots leave the other candidates blank, which counts as 0, so λ = 1 scores only the favourite. Plurality, veto and the TTR mark already name a single candidate. The Inverse-\* methods are never lazy. The lazy ballot is the voter's honest ballot: what non-strategic voters cast, round 0, and the base of VSE and WME. Strategic voters may rank more or fewer when it pays; kind voters only for a strict gain, so laziness persists unless strategy rewards completing the ranking.

**Two-round elections** (TTR, Approval runoff, Approval cond. runoff) have a real, separate runoff: voters come back and vote between the two finalists, independently of their first vote. With two candidates, voting for the preferred one is always best, so the runoff is decided by everyone's sincere preference between the finalists (turnout changes between the rounds are not modelled). The ballot carries that sincere ranking as a fixed part; strategy changes only the round-1 marks. Methods that read the runoff from the same ballot (Supplementary vote, STAR, the BTR methods) are automated runoffs, where strategic voters can also misrank the finalists.

## Methods

72 built-in methods, listed by group (see Groups below), plus any add-ons (see Add-ons); all but the references and inverse groups run by default. "0–5" in the Ballot column is the score ballot with the default range; `--max-score` changes it for all of these methods at once.

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

**condorcet-based**

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

**Add-ons: methods in separate files.** A new method does not have to go into methods.py: put a Python file into the folder `addons/`, and its methods are loaded at start and used like the built-in ones, by name or group, in run.py, sweep.py and analyze.py. Delete the file to remove them; to share a method, share the file. Files starting with `_` are not loaded. `addons/_template.py` is a commented starting point; `addons/ids.py` (IDS, iterated dropped score) is a working example. [ADDONS.md](ADDONS.md) is a tutorial for writing one with little Python: a line-by-line walkthrough of IDS, and a new method (Bucklin) built step by step. A file defines:

| Name | Required | Meaning |
|---|---|---|
| `METHODS` | yes | `{name: Method(ballot, decide, feats)}` (or `Method(ballot, full=tally)` for a rule that needs the whole ballot matrix); names must not clash with existing ones. Optional `settings={"max_score": X}` fixes the score range for this method, named `Name-X` (see Score range) |
| `FEATURES` | no | new per-voter ballot features, `{name: (function(B) -> array per voter, shape of the sum given C)}`; they are summed over voters and poll samples, which keeps tallies fast |
| `BALLOTS` | no | new ballot types, `{name: ballots.Ballot(...)}`: honest ballots, strategic changes, exaggerations, implied values and the worst ballot (see ballots.py; rarely needed) |
| `GROUP` | no | the group its methods join: an existing one (e.g. `rated`) or a new one; default `addons` |
| `test()` | no | its own checks (a small profile with a known winner), run by test_methods.py |

The ballot type (one of the built-in ones in ballots.py) decides the sincere ballots, laziness, the strategic moves and the measures, so a new method gets all of these for free. Results of add-on methods can be combined with existing runs like any other method (Combining runs): run only the new method with the same parameters and seed.

**Current add-ons:**

| Method | Ballot | File | Description |
|---|---|---|---|
| IDS | 0–5 | `addons/ids.py` | iterated dropped score, a score elimination: each round every ballot's lowest-rated remaining candidates are struck out (count 0; a ballot rating them all the same counts for none), the other scores stay as given; the lowest total is eliminated (ties for last split the win), until one is left. Unlike INS a ballot only changes where its own lowest remaining rating is, and it can be counted by hand. Group: rated |

**Groups.** Every method belongs to exactly one group. Groups can be used wherever methods are named (`--methods`, `--exclude` in run.py and sweep.py; `--show`, `--hide`, `--track` in analyze.py), they are the panels of the chain plots, and the report and summary.csv show each method's group.

| Group | Methods |
|---|---|
| `condorcet` | Ranked pairs, Schulze, Schulze-ER, Kemeny-Young, Copeland, Ranked Robin, Ranked Robin-SR, SICT, Condorcet-random-drop (Condorcet methods decided by the pairwise counts alone) |
| `minimax-family` | Minimax, Minimax-ER, MMWV, MMPO, MMPOWV (the minimax family; MMPO and MMPOWV are not Condorcet methods) |
| `condorcet-based` | Black, Baldwin, Baldwin-ER, Nanson, BTR-Borda, BTR-IRV, Benham, Smith//Score (Condorcet methods built on a Borda, IRV or score count) |
| `plurality-based` | Plurality, TTR, Supplementary vote, IRV, ER-IRV, IRV-3, Coombs, DSC, DAC (first choices, eliminations and coalitions of top preferences; Coombs eliminates by last places) |
| `approval-based` | Approval, Approval runoff, Approval cond. runoff, MDDA, MAMPO |
| `rated` | Score, MJ, MCA, Reciprocal score, INS, Geometric median (add-ons can join groups, see Add-ons) |
| `hybrid` | STAR, BTR-score, MARS, IC-MARS, MARS-T, ARA, ARA-Top, Score+, 321, MMWV+Score, MMWVxScore, STLR, Soft Minimax, Soft Minimax-40, Soft Minimax-L, Soft Minimax-D (ratings combined with pairwise comparisons or runoffs) |
| `positional` | Borda, Borda-ER, Dowdall |
| `references` | Random pair runoff, Anti-plurality, TRASH (benchmarks, not proposals; not run unless named, e.g. `--methods all` or `--methods references STAR`, and hidden in the plots by default) |
| `inverse` | Inverse-Plurality, Inverse-Approval, Inverse-Minimax, Inverse-IRV, Inverse-ER-IRV, Inverse-TTR, Inverse-Borda, Inverse-Score (tests of the strategy model, below; not run unless named; hidden in the plots by default) |

**Comparison sets**, selection names across groups (not plot panels): `er-pairs`, each strict method next to its equal-ranking version: IRV/ER-IRV, Borda/Borda-ER, Baldwin/Baldwin-ER, Minimax/Minimax-ER and MMWV, Schulze/Schulze-ER, Ranked Robin-SR/Ranked Robin. Equal-ranking versions equal the strict ones with honest ballots (sincere ballots are strict), so these pairs differ only in what strategic voters do with ties.

Example: `--methods condorcet STAR`, `--methods er-pairs`, `--methods all --exclude inverse`. Without `--methods`, run.py and sweep.py run every method except the groups references and inverse (benchmarks and model tests); the inverse group, too, is only run when named.

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.

## Analysis

`analyze.py` reads one or more elections CSVs. Rows with the same parameters except the seed form one **condition**; seeds are pooled as replications.

**Combining runs.** Each method's result in an election depends only on the parameters, the seed and the election number: every method gets the same random streams, whichever other methods run alongside (a test checks this). So results of separate runs can be pooled, and a new or changed method compared with an existing run, such as the example output, without rerunning the other methods: run it with the same parameters and seed, and give both files to `analyze.py`, the newer last.

```
python run.py --methods "My method" --out mine          # added to methods.py; same parameters and seed as the reference
python analyze.py reference mine --out compare            # reference: the folder of the existing run
```

**Older results files.** New columns (a parameter or a measure) are added to the results files over time. So that older files can still be read and combined with newer ones, each new column goes into `ADDED_COLUMNS` in analyze.py when it is added, with one of two entries:

- the value older runs implied, for a parameter that could not be set before. For example `lazy`: older runs had full ballots, so they are read as `lazy = 0.0` and count as that condition.
- blank, for a measure older runs did not record (e.g. `fidelity`, the Condorcet winner in the ballots). Tables show it as missing for those runs.

The terminal and the report list each older input and what was filled in or left blank. A file that lacks a column without such an entry (in `REQUIRED_COLUMNS`, or a run parameter) is too old and has to be regenerated; `analyze.py` says which columns are missing. When adding a run parameter: put it into `PARAMS` in run.py, read it in `one_election` from `p`, and add it to `ADDED_COLUMNS` with the value it had before.

| Column | Added | Older files |
|---|---|---|
| model | 2026-10-14 | read as clusters |
| noise_strength | 2026-10-15 | read as 3.0 (unused by other models) |
| field_kappa, field_salience | 2026-10-15 | read as 1.5 and 0 (unused by other models) |
| salience, salience_exponent, fixed_variance | 2026-10-15 | read as none, 1.0 and 0 (every dimension weighed alike, positions unscaled) |
| ks_alpha, ks_cut, ks_shared | 2026-10-14 | read as 1.5, 0.2 and 0 (unused by the clusters model) |
| lazy | 2026-10-13 | read as 0.0 (full ballots) |
| fidelity | 2026-10-12 | blank; their sincerity column holds the earlier measure (the report warns) |
| bcw, p_bcw | 2026-10-12 | blank |

**Renamed methods.** The results files store each method by its name. When a method is renamed, the old name goes into `RENAMED` in methods.py (old name → current name; groups in `RENAMED_GROUPS`), so older runs keep combining with newer ones: `analyze.py` reads old names as the current ones and says so in the report, and the old names still work in `--methods`, `--exclude`, `--show`, `--hide` and `--track`. This is only for a pure rename, with the same rule and the same results. A method whose rule changes gets a new name and no entry, so old and new results are never mixed. A removed method gets no entry either: its rows in old runs stay under their old name. An add-on may not use an old name.

| Old name | Current name |
|---|---|
| Top-two runoff | TTR |
| Approval runoff if needed | Approval cond. runoff |
| Minimax-wv | MMWV |
| MARA | ARA |
| condorcet-hybrid (group) | condorcet-based |

How the files are combined (the report lists what was done, and `analyze.py` prints it):
- **Same elections:** rows of the same election (same parameters, seed and election number) must describe the same electorate. If they do not, the files come from incompatible simulator versions and the analysis stops.
- **Duplicates:** a method that appears in several files keeps its rows from the last file, for the elections both have. The report says whether its results changed, which shows whether the method (or the simulator) was changed between the runs.
- **Coverage:** within a condition, only the elections every method has are used, so all comparisons stay paired. A run with fewer elections than the reference therefore shortens the whole condition to its elections; a run with another seed has no elections in common with the reference and stops the analysis.
- More elections for the same methods come from another seed (`--seed 2`); the seeds are pooled.
- A method taken from a run without `--participation` shows no values in the participation table.

**Uncertainty: paired bootstrap.** All methods face the same electorates, so each bootstrap replicate redraws whole elections with replacement and recomputes every method on that same draw (1000 replicates by default). This keeps comparisons paired: two methods can differ clearly even when the interval of each value on its own is wide, because elections differ much more from each other than methods differ within an election.

**The front: VSE\*, WME\* and sincerity.** What counts is the result with strategic voters and how honest they can afford to be. Honest VSE and WME depend more on modelling assumptions (who votes how sincerely is itself a model) and are reported for context, not for dominance. Above each condition's table the report gives two sets. The **Pareto front**: methods no other method matches or beats on all three point estimates (at least as good on all, better on one), however small the difference. **Undominated**: methods no other method clearly beats (see Dominated by, below), so noise-sized differences count as ties. The two need not agree: a tiny advantage keeps a method on the Pareto front but does not protect it from a method clearly better on another axis. summary.csv marks the Pareto front (`pareto_front`). The table is sorted by VSE\*.

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

**Plots** (in the analysis folder):

| File | Shows |
|---|---|
| `arrows-<c>.png` | VSE/WME plane (everything outside the reachable region greyed out), every method's shift from honest (open) to learned ballots (filled, labelled) in one picture; filled points coloured by sincerity (blue = sincere, through purple to dark red = less honest; the scale is capped at the 10th percentile of the plotted methods); faint dashed lines 1, 2, 4 and 8 points inside the frontier |
| `shift-<c>.png` | the same shifts as one row per method, VSE and WME panels; easier to read per method |
| `sincerity-<c>.png` | VSE* and WME* against sincerity |
| `fidelity-<c>.png` | every method's ballots from honest (open, at sincerity 100 and the fidelity its format allows) to learned (filled) in the fidelity / sincerity plane: going left is information lost to strategy (exaggeration, levelling), going down is preferences reversed; only for CSVs with fidelity |
| `tradeoff-<c>.png` | ternary map of the Pareto front: each point of the triangle is a weighting of VSE\*, WME\* and sincerity (corners: all weight on one axis), coloured by the Pareto method with the highest weighted sum; each axis stretched to the range the Pareto methods span on it (shown at the corners), so it shows the trade-offs within the front, not their absolute size. Pareto methods best for no weighting (in a dent of the front) are listed below. Methods hidden with `--hide` (or not in `--show`) are left out before the front is found |
| `axes-heatmap.png` | VSE\*, WME\*, sincerity, VSE and WME, methods × conditions (when there are several; methods present in every condition, sorted by VSE\*) |
| `track-<method>.png` | with `--track`: one method across all conditions in the VSE*/WME* plane (learned ballots filled, honest open). One varied parameter: a chain in its order; two (a grid, e.g. `--aggressive` × `--rounds`): lines along the parameter with more values, colour for the other; more: points with condition tags |
| `chain-<parameter>.png` | sweeps: for each parameter varied around the base (the first condition), each method's learned values as a chain of arrows through the parameter's values, light to dark; one panel per method group, all on the same scale. If the parameter only affects voters' strategy (strategic and aggressive shares, poll rounds, polls, group size), the electorates are the same and each chain starts at the method's honest point; otherwise the honest values form their own dashed chain |

Sweep plots appear automatically for every parameter that has at least two conditions differing from the base only in that parameter: a one-at-a-time sweep (`sweep.py --one-at-a-time`) or a grid over a single parameter. They show point estimates only; with few elections the zigzags in a chain are mostly noise. One far-off method (often Plurality) sets the shared scale for all panels; `--hide` or `--show` zooms in.

Methods are identified by labels, not colour. Random pair runoff, Anti-plurality and TRASH are left out of the scatter plots by default (`--hide`), since their distance compresses the scale; they stay in tables and heatmaps. `--hide` with no names shows all.

Sample size: with 160 elections per condition the intervals of single values are about ±3 points and many paired differences are still ties. For conclusions, several hundred elections per condition and at least two seeds are advisable.
