# Voting Pareto Lab

A simulator for comparing single-winner voting methods: plurality, runoffs, IRV, Condorcet methods, score and approval methods and their hybrids, 62 in all (one of them, IDS, as an add-on; more can be added), plus three reference methods and eight test methods. It asks how good the winner of each method is, not only when everyone votes honestly but also when voters respond strategically to polls, and how honest voters can afford to be under each method.

**The question.** Every voting method turns ballots into a winner, and every method can be gamed to some degree. Comparisons with honest ballots alone flatter methods that are easy to manipulate; comparisons under fully rational strategy depend on assumptions nobody believes. This lab places voters in between: some of them read polls and adjust their ballots when it pays, the way people vote tactically in real elections, and it measures what the method makes of that.

**How a simulation runs.**

1. **An electorate.** Voters and candidates are points in an issue space (by default six issues, with voters gathered in clumps within clumps). A voter likes a candidate less the farther away they are. Each election draws a new electorate ([model](docs/model.md), Electorates).
2. **Honest ballots.** Each voter fills in the method's ballot sincerely: a name, a ranking, approvals or scores, depending on the method ([model](docs/model.md), Ballots and sincere voting).
3. **Poll rounds.** Polls show the method's own counts from a sample of the electorate. Strategic voters (by default half the electorate), in groups with the same preferences, change their ballot if that improves the result for them in the polls; all groups answer the same poll at once, and the next poll shows the new ballots ([model](docs/model.md), Strategy engine).
4. **Optionally**, candidates may withdraw or field clones beforehand (Strategic nomination), and a diagnostic checks whether some voters would do better by staying home (Participation; both in [model](docs/model.md)).

**What is measured.** Two efficiencies, both 0 for a random ballot and 100 for the best candidate of each election ([analysis](docs/analysis.md), Axes):

- **VSE**, voter satisfaction efficiency: the winner's total utility (how well the electorate as a whole likes it);
- **WME**, worst-margin efficiency: the winner's smallest head-to-head margin against any rival (how strongly a majority could object that another candidate should have won).

The two pull apart, so no method can reach 100 on both; the lab computes the boundary that any rule could reach knowing everyone's true preferences (the reachable frontier). Each value comes with sincere ballots (VSE, WME) and after the poll rounds (VSE\*, WME\*), together with the **sincerity** of the strategic ballots. The main comparison is on VSE\*, WME\* and sincerity: how good the result is with strategic voters, and how honest they can be.

**What you get.** `run.py` and `sweep.py` produce CSV files with one row per election, method and ballot set; `analyze.py` turns them into a Markdown report with paired bootstrap intervals, the methods no other method beats (the front), paired comparisons and plots: how strategy moves each method in the VSE/WME plane, how methods change across parameter sweeps, and where they stand relative to the frontier.

The model builds on voter satisfaction efficiency (Jameson Quinn's VSE simulations) and on Chris Smith's spatial voter model; in the strategy model, rounds of best responses to the previous poll (level-k reasoning), the leader rule of Myerson and Weber (1993) emerges instead of being coded. Its assumptions are spelled out in [the model](docs/model.md), and eight Inverse-\* methods (honest voting elects the worst candidate) test whether voters learn their way back.

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

Every run or sweep writes into its own folder in `output/`, and `analyze.py` writes its report (`report.md`, tables and plots) into an `analysis/` folder inside it. 200 elections with all methods at the defaults take about 20 minutes; use `--methods` or `--exclude` to run fewer.

## Documentation

| File | Contents |
|---|---|
| [docs/model.md](docs/model.md) | The model: electorates and their defaults, ballots and sincere voting, the strategy engine, coordination, starting information, partial strategy, score range, strategic nomination, participation |
| [docs/methods.md](docs/methods.md) | The methods and their groups; add-ons in short |
| [docs/analysis.md](docs/analysis.md) | Measures and analysis: the axes (VSE, WME, CE), the reachable frontier, sincerity and fidelity, the front, the report and its plots, robustness and rank stability |
| [docs/running.md](docs/running.md) | Running: output folders, progress and resuming, all parameters of `run.py`, `sweep.py` and `analyze.py`, combining runs, older results files, renamed methods, the files of the code, checking changes |
| [docs/addons.md](docs/addons.md) | Tutorial: writing your own method as an add-on |
| [docs/journal.md](docs/journal.md) | Development journal: how the lab developed, the experiments, what they showed and the decisions they led to, with the settings and commands |
