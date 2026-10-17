# Voting Pareto Lab

A simulator for comparing single-winner voting methods: plurality, runoffs, IRV, Condorcet methods, score and approval methods and their hybrids, 63 in all (one of them, IDS, as an add-on; more can be added), plus three reference methods and eight test methods. A standard set of 29, the most discussed and the strongest, is what a run uses unless told otherwise. It asks how good the winner of each method is, not only when everyone votes honestly but also when voters respond strategically to polls, and how honest voters can afford to be under each method.

**The question.** Every voting method turns ballots into a winner, and every method can be gamed to some degree. Comparisons with honest ballots alone flatter methods that are easy to manipulate; comparisons under fully rational strategy depend on assumptions nobody believes. This lab places voters in between: some of them read polls and adjust their ballots when it pays, the way people vote tactically in real elections, and it measures what the method makes of that.

**How a simulation runs.**

1. **An electorate.** Voters and candidates are points in an issue space (by default six issues, with voters gathered in clumps within clumps). A voter likes a candidate less the farther away they are. Each election draws a new electorate ([model](docs/model.md), Electorates).
2. **Honest ballots.** Each voter fills in the method's ballot sincerely: a name, a ranking, approvals or scores, depending on the method ([model](docs/model.md), Ballots and sincere voting).
3. **Poll rounds.** Polls show the method's own counts from a sample of the electorate. Strategic voters (by default half the electorate), in groups with the same preferences, change their ballot if that improves the result for them in the polls; all groups answer the same poll at once, and the next poll shows the new ballots ([model](docs/model.md), Strategy engine).
4. **Optionally**, candidates may withdraw or field clones beforehand (Strategic nomination), and a diagnostic checks whether some voters would do better by staying home (Participation; both in [model](docs/model.md)).

**What is measured.** Two efficiencies, both 0 for a random ballot and 100 for the best candidate of each election ([analysis](docs/analysis.md), Axes):

- **VSE**, voter satisfaction efficiency: the winner's total utility (how well the electorate as a whole likes them);
- **WME**, worst-margin efficiency: the winner's smallest head-to-head margin against any rival (how strongly a majority could object that another candidate should have won).

The two pull apart, so no method can reach 100 on both; the lab computes the boundary that any rule could reach knowing everyone's true preferences (the reachable frontier). Each value comes with sincere ballots (VSE, WME) and after the poll rounds (VSE\*, WME\*), together with the **sincerity** of the strategic ballots. The main comparison is on VSE\*, WME\* and sincerity: how good the result is with strategic voters, and how honest they can be.

**What you get.** `run.py` and `sweep.py` produce CSV files with one row per election, method and ballot set; `analyze.py` turns them into a Markdown report with paired bootstrap intervals, the methods no other method beats (the front), paired comparisons and plots: how strategy moves each method in the VSE/WME plane, how methods change across parameter sweeps, and where they stand relative to the frontier.

The model builds on voter satisfaction efficiency (Jameson Quinn's VSE simulations) and on Chris Smith's spatial voter model; in the strategy model, rounds of best responses to the previous poll (level-k reasoning), the leader rule of Myerson and Weber (1993) emerges instead of being coded. Its assumptions are spelled out in [the model](docs/model.md), and eight Inverse-\* methods (honest voting elects the worst candidate) test whether voters learn their way back.

## Features

**Methods** ([methods](docs/methods.md))
- 63 single-winner methods in eight families: plurality-based (Plurality, TTR, IRV and variants, coalition methods), Condorcet (Ranked pairs, Schulze, Kemeny-Young, Copeland, Ranked Robin, …), the minimax family, Condorcet-based hybrids (Black, Baldwin, Nanson, BTR, Smith//Score, …), approval-based (Approval, approval runoffs, MDDA, MAMPO), rated (Score, MJ, MAV, MCA, …), score–pairwise hybrids (STAR, MARS, ARA, 3-2-1, Soft Minimax, …) and positional (Borda, Dowdall). Ties are counted exactly, as shared winning chances.
- A standard set of 29, the most discussed methods and the strongest found here, is the default; every other method, family or set can be named.
- New methods as add-ons: one Python file in `addons/` with a counting rule; ballots, strategy and measures come with the ballot type. A [tutorial](docs/addons.md) walks through writing one with little Python.

**Electorates** ([model](docs/model.md), Electorates)
- Five electorate models: fractal-noise clumps (the default, tuned against real data: Condorcet winners as often as in real ranked elections, an approval map like that of real ballots), voter clusters with Zipf dimensions and per-group salience, scale-free opinion fields, and Quinn's hierarchical model (as the lab's version and as in vse-sim's code).
- `show_electorates.py` draws and measures the electorates without any method: example plots, dimensionality, how costly the Condorcet winner is, and the candidates' map from approval ballots compared with real data.
- Candidates can be made unlike the electorate (clustered in the centre, spread out or shifted).

**Voters and strategy** ([model](docs/model.md), Strategy engine)
- Poll rounds: voters see only the method's own poll counts, with realistic poll error, and answer them all at once; groups with the same preferences act as blocs. Aggressive voters exaggerate whenever it cannot hurt, kind ones only for a clear gain.
- Who strategises: a fixed share of strategic voters, or adaptive voters who strategise in a round when the race between the two leaders matters enough to them, so attack and defence come from the same rule.
- Laziness: truncated rankings and fewer approvals or scores, each voter with their own cut.
- Also: convergent learning as a long-run alternative, strategic nomination (candidates withdraw or field clones), a first poll from a previous election (does a method let the electorate leave a strategic state?), and a check of the no-show paradox.

**Measures** ([analysis](docs/analysis.md))
- VSE and WME with the same reference points (random ballot and best candidate), with honest and with strategic ballots, plus Condorcet efficiency.
- The reachable frontier and region: what any rule could reach knowing every voter's utilities.
- Sincerity (how far strategic ballots go against the voters' own preferences, scaled per ballot format) and fidelity (how much of their preferences the ballots convey), the kinds of strategy used (compromise, burial, push-over), and the Condorcet winner as cast ballots would show it.

**Analysis** ([analysis](docs/analysis.md), Analysis)
- Paired bootstrap over elections, so methods are compared on the same electorates; the Pareto front and the methods no other clearly beats; paired differences.
- Sweeps over any parameter, one at a time or as a grid, with rank stability against a noise reference: does an assumption change which methods do well?
- Plots: honest-to-strategic shifts in the VSE/WME plane, sincerity and fidelity, a ternary map of the front, chains across sweeps, heatmaps and single methods tracked across conditions.

**Working with runs** ([running](docs/running.md))
- One folder per run with its analysis; runs can be resumed after an interruption and give the same result row for row; results of separate runs combine, so a new method is compared without rerunning the others; older results files stay readable.
- Hand-built tests and a regression check that compares every number with a stored reference after a code change.

## Getting started

Python 3.9 or newer + NumPy; the analysis also needs matplotlib (`pip install -r requirements.txt`).

```
python test_methods.py                                   # hand-built checks
python regression.py                                     # same numbers as before? (after a code change)
python run.py --out base                                 # one condition at the defaults -> output/base/
python sweep.py --rounds 1 2 3 --aggressive 0 0.5 1 --out rounds   # grid of conditions -> output/rounds/
python analyze.py rounds --compare "BTR-score:STAR"      # -> output/rounds/analysis/
python show_electorates.py --model noise clusters --out look        # look at the electorates, no methods
```

Every run or sweep writes into its own folder in `output/`, and `analyze.py` writes its report (`report.md`, tables and plots) into an `analysis/` folder inside it. A run uses the standard set of 29 methods; `--methods all` runs every one (200 elections with all methods take about 20 minutes), and `--methods` takes methods, groups or sets by name.

## Documentation

| File | Contents |
|---|---|
| [docs/model.md](docs/model.md) | The model: electorates and their defaults, ballots and sincere voting (with laziness and the score range), the strategy engine (poll rounds, partial strategy and adaptive voters, convergent learning), starting information, strategic nomination, participation |
| [docs/methods.md](docs/methods.md) | Groups and sets, every method's rule, add-ons |
| [docs/analysis.md](docs/analysis.md) | Measures and analysis: the axes (VSE, WME, CE), the reachable frontier, strategy types, sincerity and fidelity, the front, the report and its plots, robustness and rank stability |
| [docs/running.md](docs/running.md) | Running: output folders and what a run writes, progress and resuming, all parameters of `run.py`, `sweep.py` and `analyze.py`, combining runs, older results files, renamed methods, the files of the code, checking changes |
| [docs/addons.md](docs/addons.md) | Tutorial: writing your own method as an add-on |
| [docs/journal.md](docs/journal.md) | Development journal: how the lab developed, the experiments, what they showed and the decisions they led to, with the settings and commands |
