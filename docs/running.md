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

`--out` names the folder; a name without a directory goes into `output/` (created when needed), one with a directory (`--out runs/test`) is used as given. Without `--out` the folder is named by the time (`run-20261011-143000`, `sweep-…`), so nothing is overwritten by accident; an existing run is only replaced with `--overwrite`. `analyze.py` takes run or sweep folders (a sweep folder means all its run files) or elections CSV files, and looks up bare names in `output/`. Since a sweep keeps one file per run, single runs can be analysed alone, moved, or mixed with runs of other sweeps (see Combining runs). With several inputs (combining runs) the first is the base and the analysis goes into the last input's folder as `analysis-vs-<first>/`: `analyze.py base maxscore` writes `output/maxscore/analysis-vs-base/`, so each test keeps its comparison with the base next to its data. `--out` names another folder; a run or sweep folder given there gets the analysis inside it, never over its own files.

**What a run writes.** `elections.csv` has one row per (election, method, ballots = sincere or learned) with the run's parameters as columns; the simulator version (column `version`, from `run.VERSION`; the report names it, and warns when CSVs of different versions are combined); the quantities for the axes: U_method, U_best, U_rb (random ballot), U_rand (random candidate, for the usual VSE), WM (the winner's worst margin, share of voters), WM_best, WM_rb; has_cw and p_cw (probability of electing the Condorcet winner); the electorate properties cw_is_util, U_cw and fav_ratio (the same in every row of an election; see [analysis.md](analysis.md), Robustness); sincerity, fidelity, settled, the strategy types, the ties (split, zero_margins; see [analysis.md](analysis.md), Ties) and, with `--adaptive`, adapted. `summary.csv` and the printed table (`table.txt`) have one row per method:

| Column | Meaning |
|---|---|
| VSE, WME | voter satisfaction and worst-margin efficiency, sincere ballots |
| VSE_learned, WME_learned | the same with learned ballots |
| CE, CE_learned | Condorcet efficiency, sincere and learned ballots |
| sincerity | mean sincerity of the learned ballots, over all voters ([analysis.md](analysis.md), Sincerity and fidelity; elections.csv also has `fidelity`, for both ballot sets) |
| settled | poll rounds: share of elections where the last round gave the same ballots as the round before (a fixed point); convergent learning: no group changed its ballot in the last round. Cycles never settle |

`on_front` in the summary is a quick check on point estimates (VSE_learned, WME_learned, sincerity; no uncertainty). The analysis uses the tie-aware "Dominated by" instead ([analysis.md](analysis.md), Analysis).

**Progress, saving and resuming.** While a run or sweep works, a line shows the elections done, the time so far and an estimate of the time left (redrawn in place after every election in a terminal; when the output goes to a file, a new line at every tenth of the way). Each election is written to `elections.csv` as soon as it is done, so nothing finished is lost, and a run in progress can already be analysed. If a run stops (Ctrl-C, a crash, the computer shut down), the same command with `--resume` (and the same `--out`) continues it: complete elections are kept, an incomplete last one is dropped, and only the missing ones are run. Since every election depends only on its parameters, seed and number, the result is identical to an uninterrupted run, row for row. After Ctrl-C the exact command to continue is printed. A larger `--elections` with `--resume` extends a finished run the same way (the first elections are kept, only the new ones are run). `--resume` stops if the command differs otherwise (other methods, parameters or simulator version); this includes methods that joined the selection since, e.g. a new add-on with `--methods all` (until 2026-10-19 that case silently started the run over). A sweep resumes each of its runs (in `runs/`) the same way.

200 elections with all methods (`--methods all --exclude inverse`) at the default settings of the time (three poll rounds, 10 polls per round) took about 20 minutes in one measurement, and 5 polls per round (the default since 2026-10-26) take about 60% of that; time scales linearly with cores, with the number of poll rounds and with the polls per round. To try a new method, there is no need to rerun the others: see Combining runs. More candidates and more voters take longer: one election with all methods took about 50 s with 9 candidates, 70 s with 10, 130 s with 11 and 180 s with 12 in one measurement (one core per election), the slowest methods there being Ranked pairs, DAC and DSC. Several methods enumerate subsets of candidates, so beyond 12 the time grows quickly. Use `--methods` or `--exclude` to run fewer, by name or by group (see [methods.md](methods.md), Methods).

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
| `--cluster-zipf` | 0 | model `clusters`: 1 = dimension i has scale 1/i ([model.md](model.md), Zipf dimensions); 0 = all dimensions equal |
| `--cluster-count` | 20 | model `clusters`: voter clusters |
| `--cluster-spread` | 0.5 | model `clusters`: standard deviation of voters around their cluster centre (centres have 1), before the Zipf scaling |
| `--cluster-salience-exponent` | 0.5 | model `clusters`: each cluster weighs the issues 1/i^s in its own random order (1 = Zipf; 0 = every issue alike, no salience; see [model.md](model.md), Electorates) |
| `--cluster-fixed-variance` | 1 | model `clusters`: 1 = scale positions so the dimensions' total variance is π²/6 whatever their number |
| `--field-kappa` | 1.5 | model `field`: saturation of opinions (small: smooth, nearly Gaussian; large: two camps per issue) |
| `--field-salience` | 0 | model `field`: strength of the salience field (0 = every issue weighed alike) |
| `--ks-alpha` | 1.5 | model `ks`: faction alpha (higher: more, smaller factions); see [model.md](model.md), Electorates |
| `--ks-cut` | 0.2 | model `ks`: importance below which no further issue bundles or dimensions are drawn (lower: more dimensions) |
| `--ks-shared` | 0 | model `ks`: 1 = the same factions in every issue bundle (a test variant without cross-cutting cleavages) |
| `--width` | 0.7 | distance scale of the utility fall-off |
| `--utility` | `exp` | utility of distance: `exp` = exp(−distance / width), `linear` = −distance / width, `quadratic` = −(distance / width)² (see [model.md](model.md)) |
| `--cand-dispersion` | 1.0 | candidates' distance from the electorate's centre × this (< 1 clustered, > 1 spread out); see [model.md](model.md), Candidates unlike the electorate |
| `--cand-shift` | 0.0 | candidates moved along the main dimension by this many voter SDs; see [model.md](model.md), Candidates unlike the electorate |
| `--seed` | 1 | random seed; election e uses the stream (seed, e), so all methods face the same electorates |
| `--rounds` | 3 | poll rounds before the vote (0 = everyone sincere); see [model.md](model.md), Strategy engine. Three rounds take voters past the first answer to the honest poll, which leans on the modelling of sincere ballots, toward the state they settle in |
| `--samples` | 5 | polls per round (they stand for the uncertainty; 10 until 2026-10-26); with `--converge`, bootstrap resamples per round |
| `--poll-error` | 2 | voters treat polls as this many times less precise than their sampling error |
| `--aggressive` | 0.5 | share of strategic voters who exaggerate whenever it cannot hurt; the others (kind) only when the polls show a gain (poll rounds only) |
| `--campaign` | 0 (off) | poll rounds, experimental: above 0, every candidate recommends a move each round, planned with this share of its audience following (0.5 suggested); bold (aggressive) voters may follow ([campaigns.md](campaigns.md)) |
| `--risk` | 0 | poll rounds, experimental: 1 = each voter is careful or bold by their satisfaction with the polls and a personal threshold, instead of the fixed `--aggressive` split ([campaigns.md](campaigns.md)) |
| `--risk-tolerance` | 0.1 | with `--risk 1`: how much loss bold voters accept, per unit of their shortfall below their threshold |
| `--lazy` | 0 | laziness of honest ballots, 0–1: each voter's cut is drawn between 0 and this (see [model.md](model.md), Truncation and laziness) |
| `--poll-size` | 0 | voters per poll; 0 = half the electorate, rounded to an odd number |
| `--search` | 10 | random ballot changes a group tries per poll round, after the naive exaggerations |
| `--converge` | 0 | N > 0: convergent learning with N rounds instead of poll rounds (e.g. `--converge 20 --samples 10`) |
| `--group-size` | 10 | convergent learning only: most voters that coordinate as one group: a number of voters (1 = every voter alone), a share of the electorate between 0 and 1 (0.05 = 5% of the voters, rounded, at least 1), or 0 = whole ranking group |
| `--strategic` | 0 | share of voters who always vote strategically (see [model.md](model.md), Partial strategy) |
| `--adaptive` | the rest | share of voters who decide each poll round whether to vote strategically, by how much the race between the two leaders matters to them (poll rounds only; `--strategic` + `--adaptive` ≤ 1, sweep.py skips other combinations; see [model.md](model.md), Partial strategy); left out: everyone not in `--strategic` (1 − `--strategic`), with `--converge` 0 |
| `--max-score` | 5 | score ballots range over 0–max (see [model.md](model.md), Score range) |
| `--exits` | 0 | 1: candidates may withdraw and re-enter strategically (see [model.md](model.md), Strategic nomination) |
| `--clones` | 0 | most clones in the race at once, fielded strategically; 0 = none (see [model.md](model.md), Strategic nomination) |
| `--start` | honest | what the first poll shows: `honest` ballots, or `previous`: the result of a previous election (see [model.md](model.md), Starting information) |
| `--participation` | off | also check participation (no-show paradox) on sincere and learned ballots (see [model.md](model.md), Participation); slower |
| `--manipulation` | off | also test coalition manipulation from honest ballots, one- and two-sided (see [model.md](model.md), Manipulation test) |
| `--workers` | all cores | parallel processes |
| `--methods` | the `standard` set (29 methods) | run only these methods (start from none); method, group or set names (see [methods.md](methods.md), Groups), case ignored; `all` for every method |
| `--exclude` | none | leave these out (from the default, or from `--methods`) |
| `--out` | time-stamped | folder of the run: `output/<out>/` (see One folder per run) |
| `--overwrite` | off | replace an existing run in `--out` |
| `--resume` | off | continue an interrupted run in `--out` with the same command (see Progress, saving and resuming) |

`sweep.py` takes the same flags, but every condition flag (`--voters` … `--width`, `--cluster-zipf`, `--rounds`, `--samples`, `--poll-size`, `--poll-error`, `--aggressive`, `--search`, `--converge`, `--group-size`, `--strategic`, `--adaptive`, `--max-score`, `--exits`, `--clones`) accepts several values, and `--seeds` several seeds; it runs the full grid, one run after another, and writes one elections CSV per run into `output/<out>/runs/`. With `--one-at-a-time`, the first value of every flag is the base, and each other value is run with everything else at the base, instead of the full grid. Conditions that differ only in `--rounds` come from one run: the poll rounds up to the highest are computed once and each requested round is written as its own condition (round r of a longer run is exactly what `--rounds r` gives), so `--rounds 1 2 3 4 5` costs as much as `--rounds 5`.

`analyze.py`:

| Flag | Default | Meaning |
|---|---|---|
| (positional) | | one or more run or sweep folders (or elections CSV files) |
| `--out` | `analysis/` in the input folder; with several inputs `analysis-vs-<first>/` in the last one | analysis folder (a run or sweep folder named here gets the analysis inside it) |
| `--boot` | 1000 | bootstrap replicates |
| `--format` | png | file format(s) of the plots, one or more of png, svg, pdf (`--format png svg` writes both); the report links the first. svg and pdf are vector graphics, sharp at any zoom |
| `--dpi` | 150 | resolution of png plots |
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
python analyze.py reference mine                         # -> output/mine/analysis-vs-reference/
```

**Older results files.** New columns (a parameter or a measure) are added to the results files over time. So that older files can still be read and combined with newer ones, each new column goes into `ADDED_COLUMNS` in analyze.py when it is added, with one of two entries:

- the value older runs implied, for a parameter that could not be set before. For example `lazy`: older runs had full ballots, so they are read as `lazy = 0.0` and count as that condition.
- blank, for a measure older runs did not record (e.g. `fidelity`, the Condorcet winner in the ballots). Tables show it as missing for those runs.

The terminal and the report list each older input and what was filled in or left blank. A file that lacks a column without such an entry (in `REQUIRED_COLUMNS`, or a run parameter) is too old and has to be regenerated; `analyze.py` says which columns are missing. When adding a run parameter: put it into `PARAMS` in run.py, read it in `one_election` from `p`, and add it to `ADDED_COLUMNS` with the value it had before.

Column names are the option names with underscores, except for the options of the clusters model, renamed on 2026-10-17, whose columns kept their earlier names so older files stay readable: `--cluster-count` → `clusters`, `--cluster-spread` → `spread`, `--cluster-zipf` → `zipf`, `--cluster-salience-exponent` → `salience_exponent`, `--cluster-fixed-variance` → `fixed_variance`. The column `salience` (`cluster` or `none`) is filled in from the exponent (above 0 or 0).

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
| split, zero_margins | 2026-10-26 | blank (ties not recorded) |
| U_ml, WM_ml | 2026-10-26 | blank (no maximal lottery reference point) |
| campaign, risk, risk_tolerance; camp_follow, camp_top, bold | 2026-10-27 | read as 0, 0 and 0.1 (no campaigns, the fixed aggressive split); the three outputs blank |
| manip1, manip2, U_manip1, U_manip2 | 2026-10-17 | blank (manipulation not tested) |

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

## Safety

- **Output folders.** An existing folder given as `--out` must hold only what the lab writes there: for `run.py` its results files (`elections.csv`, `summary.csv`, `table.txt`, `command.txt`) and `analysis*/` folders, for `sweep.py` `runs/` and `command.txt`, for `analyze.py` a report with its tables and plots, for `show_electorates.py` its plots and `summary.txt`, for `compare.py` its report and tables. Anything else (hidden files aside) and the script stops before writing, so a mistyped `--out` (a home or project folder) is refused instead of filled or overwritten. An existing run is replaced only with `--overwrite`; its old `analysis*/` folders are then out of date, which the script says.
- **Sweeps over earlier sweeps.** If `runs/` of the folder holds run files the new sweep does not write, it stops: `analyze.py` would read them as part of the sweep. Use another `--out`, or move them out.
- **Nothing is deleted** beyond what is asked for: no script removes files or folders. `--overwrite` replaces the results files of that one run. `--resume` drops the unfinished last elections and rewrites `elections.csv` through a temporary file, so an interruption leaves either the old or the new file. The tests make their own temporary folders and remove only those.
- **Fingerprints.** The version number stays the same while the model and the methods are developed, so it cannot tell runs of older code from newer ones. `run.py` and `sweep.py` therefore record, per method, a fingerprint of what the code computes: the method's results in a small fixed test (6 elections, 31 voters, 6 candidates, some voters strategic and some adaptive; since 2026-10-26 also 6 elections made for ties: 20 voters, 5 candidates, laziness 0.5, where every method on ordered ballots meets tied pairs), hashed (column `fingerprint`, a few seconds per run for the standard set). The same fingerprint means the same results as when the other run was made; a different one means the code changed in between. `--resume` refuses to continue a run whose methods now have other fingerprints; `analyze.py` notes when it pools a method with different fingerprints; `compare.py` refuses it. A change that only shows under other settings (another electorate model, nomination, linear utility), or in none of the test elections, is not seen; a new tie rule is seen only if a tie decides one of them.
- **Add-ons are programs** that run whenever the lab starts: add only files you trust (see [methods.md](methods.md), Add-ons).
- **`compare.py`** reads only the folder given, refuses folders with other data in them, and refuses mixed simulator versions unless forced (see below).

## Comparing across many runs

`analyze.py` compares methods within each condition of its inputs. `compare.py` looks across all of them: how a method does over every condition in a folder, and how often one method beats another, with the exceptions.

```
python compare.py ../bffl-runs                      # -> ../bffl-runs/comparison/
python compare.py output/falloff --pairs "ESR:Score" "Score runoff:Minimax"
```

- **Input:** one folder, searched for run and sweep folders (with `elections.csv` or `runs/`) at most `--depth` levels down (default 2: `FOLDER/test/` and `FOLDER/group/test/`); everything found is read, nothing outside it. Collect the runs that belong together in one folder (the runs repository, or a subfolder of `output/`), and leave exploratory or outdated runs out.
- **Precautions** against a wrong folder (a home or system folder): any subfolder without run data stops the search, with a list of them, before anything is read; hidden folders (`.git`), symbolic links and the lab's report folders (`analysis*/`, `overview-*/`, `comparison/`) are skipped. The output folder must be new, empty or an earlier comparison; a folder with other files is refused, so nothing else is overwritten.
- **Versions and fingerprints:** all inputs must come from the same simulator version, and every method must have the same fingerprint in all of them (see Safety); otherwise the comparison stops and lists which input has which. `--force-versions` compares them anyway and says so at the top of the report. Files from before fingerprints (2026-10-23) cannot be checked; the report lists them.
- **Conditions:** one condition is one set of run parameters. Results files that share elections are pooled (runs of different methods on the same elections). When one has only some of the other's elections (fewer elections with the same seed), the pooled methods are compared on the elections all of them have, and the report says how many were used; as separate replicates the shared elections would count twice (until 2026-10-26 they did). Files with no election in common under the same parameters (another seed) are analysed as replicates. **Every condition counts once**, for every method run in it, its replicates sharing that weight; a parameter swept in many runs therefore does not outweigh one tested once. Counts can then be fractional (2.5 of 4 conditions).
- **Within a condition** the comparisons are those of `analyze.py`, with the same bootstrap draws (they depend only on the condition's elections and `--seed`, so the comparison agrees with each run's own report, and adding runs leaves the other conditions' results unchanged): paired bootstrap intervals, clearly better when the interval excludes 0 (sincerity: and at least 1 point), dominance on VSE\*, WME\* and sincerity. **Near the front (1 / 2):** no method in the condition is better by 1 (2) points on VSE\* and WME\* at once (point estimates), the guide lines of the arrow plots.
- **The report** (`comparison/report.md`): the conditions, named by what differs from the defaults of `run.py`; per method the share of its conditions in which it is undominated, on the Pareto front and near the front, its means and its worst condition; a matrix of who dominates whom among the leading methods (`--top`, default 12; only methods run in at least half the conditions, `--min-conditions`, the others are listed separately); and the **exceptions**: for pairs of leading methods where each is ahead somewhere, how often each is, and in which conditions the less usual outcome happens, with the settings those conditions share. Pairs asked for with `--pairs "A:B"` get a section of their own (**Requested pairs**, in the order given), each with a line also when there is no clear difference, when the two were never run together, or when one is not among the compared methods. The names are checked before any data is read; a misspelled one stops the comparison with the closest names as suggestions.
- **Progress and memory:** the terminal shows the three steps (reading, analysing each condition, the report), each file and condition as it is done, and every million rows of a large file. Each file is read once and only the columns the comparison needs are kept, as compact numbers (about 70 bytes per row instead of every column), so several gigabytes of results fit in a few hundred megabytes of memory; in a test, 600,000 rows (0.34 GB) took 12 seconds and 350 MB.
- **Clear dominance (beatpath):** `beatpath.png` draws the methods of the main table in layers, with an arrow from A down to B when A dominates B in at least half of the conditions both were run in (`--beat-share`) and more often than B dominates A. Arrows implied by a chain of others are left out, so the top layer holds the methods nobody dominates at that level, and each method sits below the lowest one that dominates it. The report lists the arrows with their shares.
- **Tables:** `methods.csv`, `pairs.csv` (every ordered pair: shared conditions, clearly better per axis, dominates, and in which conditions), `conditions.csv`, `beatpath.csv` (every pair at the beatpath share, and whether its arrow is drawn).
- The averages describe the collection of runs in the folder, not all possible electorates: which conditions are in it decides what "on average" means, so the report lists them all.

| Flag | Default | |
|---|---|---|
| `--out` | `FOLDER/comparison` | output folder |
| `--methods`, `--exclude` | all found except references and inverse | methods, groups or sets |
| `--pairs` | none | pairs `"A:B"` reported in detail |
| `--top` | 12 | leading methods in the pair matrix and the exceptions |
| `--min-conditions` | half of the conditions | fewer: listed separately |
| `--min-elections` | 100 | conditions (or replicates) with fewer elections are left out, and listed in the report |
| `--boot` | 1000 | bootstrap replicates per condition (as `analyze.py`, so both agree) |
| `--format` | png | file format(s) of the beatpath plot, one or more of png, svg, pdf (`--format png svg` writes both); the report links the first. svg and pdf are vector graphics, sharp at any zoom |
| `--dpi` | 150 | resolution of png plots |
| `--beat-share` | 0.5 | beatpath: an arrow when one method dominates the other in at least this share of the shared conditions |
| `--depth` | 2 | folder levels searched for runs |
| `--force-versions` | off | compare mixed simulator versions (stated in the report) |
| `--seed` | 0 | bootstrap random seed |

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
| `manipulation.py` | Manipulation test: coalitions for each candidate against the honest winner, one- and two-sided |
| `outcomes.py` | What an outcome is (winner probabilities) and what it is worth, `value(P, v)`: the one place all judging of outcomes goes through |
| `metrics.py` | Efficiencies for three baselines, winner's worst margin, Condorcet efficiency, electorate properties, Pareto front |
| `run.py` | Tallies every electorate with sincere and learned ballots, runs in parallel, writes CSV with the run's parameters as columns |
| `sweep.py` | Runs a grid of conditions (or one parameter at a time) and seeds, one CSV per run |
| `analyze.py` | Paired bootstrap, intervals, dominance pointers, paired comparisons, rank stability, Markdown report |
| `compare.py` | Across all runs in a folder: each method over every condition, who beats whom how often, and the exceptions |
| `plots.py` | The analysis plots |
| `test_methods.py` | Hand-built checks |
| `regression.py`, `regression.csv` | Regression check: a few tiny fixed runs compared with the stored reference (see Checking changes) |
| `docs/` | The documentation (listed in README.md): model, methods, analysis, running, the add-on tutorial and the development journal |

## Checking changes

Two checks, both quick:

- `python test_methods.py`: hand-built checks. Each method on small profiles with a known winner, the ballot types, truncation and laziness, sincerity and fidelity, the add-ons.
- `python regression.py`: does the simulator still give the same numbers? It runs twelve tiny conditions (two elections each: the base, five candidates, laziness with scores 0..10, convergent learning, strategic nomination, the first poll from a previous election, one for each electorate model and the salience option, and adaptive voters with laziness) with every method, including the references, the Inverse-\* methods and the add-ons, and with the manipulation test, and compares every column of every row with `regression.csv`. About a minute or less.

A change that only reorganises the code (moving, renaming, splitting files) must leave `regression.py` at "all equal". A change meant to alter results (a fixed bug, a new rule) shows up as differences, listed by method and column; once they are understood, `python regression.py --update` writes the new reference, and `regression.csv` is committed together with that change, so the history shows which commit changed which numbers. New methods (a new add-on) are listed but not counted as differences; `--update` adds them. Numbers count as equal within a relative tolerance of 1e-9, since the last digits can differ between numpy versions or machines.
