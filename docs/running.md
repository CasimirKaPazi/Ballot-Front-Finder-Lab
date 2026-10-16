# Running the simulator

How to run conditions and sweeps, what they write, and the parameters. What the model does is in [model.md](model.md); what is measured and reported in [analysis.md](analysis.md).

## Output, progress and resuming

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

200 elections with all methods at the default settings (three poll rounds, 10 polls per round) took about 20 minutes in one measurement; time scales linearly with cores, with the number of poll rounds and with the polls per round. To try a new method, there is no need to rerun the others: see Combining runs. More candidates and more voters take longer. Use `--methods` or `--exclude` to run fewer, by name or by group (see [methods.md](methods.md), Methods).

## Parameters

`run.py` (one condition). The electorate defaults are explained in [model.md](model.md), Electorates (Defaults, and why):

| Flag | Default | Meaning |
|---|---|---|
| `--elections` | 200 | electorates |
| `--voters` | 99 | voters per election (odd, so pairwise majorities rarely tie) |
| `--cands` | 6 | candidates |
| `--model` | noise | electorate model: `noise`, `clusters`, `field`, `ks` or `ks-quinn` (see [model.md](model.md), Electorates; `models.py`) |
| `--dims` | 6 | issue-space dimensions (models `noise`, `clusters`, `field`) |
| `--noise-strength` | 5 | model `noise`: how strongly voters gather where the fractal noise is high (0 = a Gaussian blob; larger: sharper clumps) |
| `--zipf` | 0 | model `clusters`: 1 = dimension i has scale 1/i ([model.md](model.md), Zipf dimensions); 0 = all dimensions equal |
| `--clusters` | 20 | model `clusters`: voter clusters |
| `--spread` | 0.5 | model `clusters`: standard deviation of voters around their cluster centre (centres have 1), before the Zipf scaling |
| `--salience` | cluster | model `clusters`: `cluster` = each cluster weighs the issues 1, 1/2, 1/3, … in its own random order (see [model.md](model.md), Electorates) |
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
| `--rounds` | 3 | poll rounds before the vote (0 = everyone sincere); see [model.md](model.md), Strategy engine. Three rounds take voters past the first answer to the honest poll, which leans on the modelling of sincere ballots, toward the state they settle in |
| `--samples` | 10 | polls per round (they stand for the uncertainty); with `--converge`, bootstrap resamples per round |
| `--poll-error` | 2 | voters treat polls as this many times less precise than their sampling error |
| `--aggressive` | 0.5 | share of strategic voters who exaggerate whenever it cannot hurt; the others (kind) only when the polls show a gain (poll rounds only) |
| `--lazy` | 0 | laziness of honest ballots, 0–1: each voter's cut is drawn between 0 and this (see [model.md](model.md), Truncation and laziness) |
| `--poll-size` | 0 | voters per poll; 0 = half the electorate, rounded to an odd number |
| `--search` | 10 | random ballot changes a group tries per poll round, after the naive exaggerations |
| `--converge` | 0 | N > 0: convergent learning with N rounds instead of poll rounds (e.g. `--converge 20 --samples 10`) |
| `--group-size` | 10 | convergent learning only: most voters that coordinate as one group: a number of voters (1 = every voter alone), a share of the electorate between 0 and 1 (0.05 = 5% of the voters, rounded, at least 1), or 0 = whole ranking group |
| `--strategic` | 0.5 | share of voters who always vote strategically (see [model.md](model.md), Partial strategy) |
| `--adaptive` | 0 | share of voters who decide each poll round whether to vote strategically, by how much the race between the two leaders matters to them (poll rounds only; `--strategic` + `--adaptive` ≤ 1, sweep.py skips other combinations; see [model.md](model.md), Partial strategy) |
| `--max-score` | 5 | score ballots range over 0–max (see [model.md](model.md), Score range) |
| `--exits` | 0 | 1: candidates may withdraw and re-enter strategically (see [model.md](model.md), Strategic nomination) |
| `--clones` | 0 | most clones in the race at once, fielded strategically; 0 = none (see [model.md](model.md), Strategic nomination) |
| `--start` | honest | what the first poll shows: `honest` ballots, or `previous`: the result of a previous election (see [model.md](model.md), Starting information) |
| `--participation` | off | also check participation (no-show paradox) on sincere and learned ballots (see [model.md](model.md), Participation); slower |
| `--workers` | all cores | parallel processes |
| `--methods` | all but references and inverse | run only these methods (start from none); method or group names (see [methods.md](methods.md), Methods), case ignored; `all` for every method |
| `--exclude` | none | leave these out (from the default, or from `--methods`) |
| `--out` | time-stamped | folder of the run: `output/<out>/` (see One folder per run) |
| `--overwrite` | off | replace an existing run in `--out` |
| `--resume` | off | continue an interrupted run in `--out` with the same command (see Progress, saving and resuming) |

`sweep.py` takes the same flags, but every condition flag (`--voters` … `--width`, `--zipf`, `--rounds`, `--samples`, `--poll-size`, `--poll-error`, `--aggressive`, `--search`, `--converge`, `--group-size`, `--strategic`, `--adaptive`, `--max-score`, `--exits`, `--clones`) accepts several values, and `--seeds` several seeds; it runs the full grid, one run after another, and writes one elections CSV per run into `output/<out>/runs/`. With `--one-at-a-time`, the first value of every flag is the base, and each other value is run with everything else at the base, instead of the full grid. Conditions that differ only in `--rounds` come from one run: the poll rounds up to the highest are computed once and each requested round is written as its own condition (round r of a longer run is exactly what `--rounds r` gives), so `--rounds 1 2 3 4 5` costs as much as `--rounds 5`.

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

## Combining runs and older files

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
| adaptive, adapted | 2026-10-16 | read as 0 (no adaptive voters); adapted blank |
| lazy | 2026-10-13 | read as 0.0 (full ballots). From 2026-10-16 each voter has their own cut between 0 and `lazy`; lazy runs from before (one cut for everyone) are read the same way, so do not combine them with newer ones |
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

## Files

| File | Contents |
|---|---|
| `models.py` | Electorate models (one entry per model in `MODELS`): positions of voters and candidates; distances and absolute utilities `exp(-distance / width)`, common to all models |
| `show_electorates.py` | Plots and numbers of the electorates a model draws, without methods: example elections, dimensionality, steepness, utilities, Condorcet winner (see [model.md](model.md), Looking at the electorates) |
| `scenario.py` | The settings of a condition's electorates (model and parameters) |
| `ballots.py` | Ballot types, one entry per type in `BALLOTS`: encoding, sincere ballots, laziness, strategic changes and exaggerations, implied values and the worst ballot (for fidelity and sincerity) |
| `methods.py` | Exact tallies on ballot matrices (ties shared), method groups, renamed methods, add-on loader |
| `addons/` | Methods in separate files, loaded at start (see [methods.md](methods.md), Add-ons); `_template.py` to start from |
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
| `docs/` | The documentation (listed in README.md): model, methods, analysis, running, the add-on tutorial and the development journal |

## Checking changes

Two checks, both quick:

- `python test_methods.py`: hand-built checks. Each method on small profiles with a known winner, the ballot types, truncation and laziness, sincerity and fidelity, the add-ons.
- `python regression.py`: does the simulator still give the same numbers? It runs twelve tiny conditions (two elections each: the base, five candidates, laziness with scores 0..10, convergent learning, strategic nomination, the first poll from a previous election, one for each electorate model and the salience option, and adaptive voters with laziness) with every method, including the references, the Inverse-\* methods and the add-ons, and compares every column of every row with `regression.csv`. About a minute or less.

A change that only reorganises the code (moving, renaming, splitting files) must leave `regression.py` at "all equal". A change meant to alter results (a fixed bug, a new rule) shows up as differences, listed by method and column; once they are understood, `python regression.py --update` writes the new reference, and `regression.csv` is committed together with that change, so the history shows which commit changed which numbers. New methods (a new add-on) are listed but not counted as differences; `--update` adds them. Numbers count as equal within a relative tolerance of 1e-9, since the last digits can differ between numpy versions or machines.
