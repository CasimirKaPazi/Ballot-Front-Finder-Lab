# Voting Pareto Lab

Python 3 + NumPy; the analysis also needs matplotlib. Honest and learned-strategy elections, evaluated on a Pareto front.

```
python test_methods.py                                   # hand-built checks
python run.py --elections 200                            # one condition, prints a table
python sweep.py --group-size 1 10 0 --seeds 1 2          # grid of conditions -> sweep-elections.csv
python analyze.py sweep-elections.csv --compare "BTR-score:STAR"   # -> analysis/
```

300 elections with all 25 methods take about 7 minutes on 2 cores; time scales linearly with cores. Use `--methods` or `--exclude` to run fewer.

## Files

| File | Contents |
|---|---|
| `scenario.py` | Electorate generator: spatial clusters, absolute utilities `exp(-distance / width)` |
| `methods.py` | Ballot types, sincere ballots, exact tallies on ballot matrices (ties shared) |
| `strategy.py` | Learning engine and cognitive dissonance |
| `metrics.py` | Efficiencies for three baselines, Condorcet efficiency, Pareto front |
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

  A group that deviates sometimes (15%) proposes going back to sincere.
- **Acceptance:** a proposal is kept only if it raises the group's expected utility.
- **Rounds:** 15, groups in random order each round.
- **Random streams:** each method learns with its own random stream per election, keyed by the method's name, so results do not change when methods are added or reordered.

## Parameters

Scenario flags as in step 1 (`--voters`, `--cands`, `--dims`, `--clusters`, `--spread`, `--width`, `--seed`), plus:

| Flag | Default | Meaning |
|---|---|---|
| `--elections` | 200 | electorates |
| `--rounds` | 15 | learning rounds |
| `--samples` | 10 | poll resamples per round |
| `--group-size` | 0 | most voters that coordinate as one group; 0 = whole ranking group, 1 = every voter alone |
| `--workers` | all cores | parallel processes |
| `--methods` | all | run only these methods (start from none); names as in the table below, case ignored |
| `--exclude` | none | run all methods except these |

## Coordination: what the strategic axes measure

How many voters move together decides which strategies pay, and the methods differ sharply in this.

- **Compromise** (voting for a viable candidate instead of a hopeless favourite) pays even for a single voter. It is the main strategy in Top-two runoff and IRV, and it tends to carry the Condorcet winner into the final round.
- **Burial** (ranking or rating a rival below a candidate one likes less, to create a cycle or to drop the rival in a runoff) only pays when many voters do it together. A single voter out of 99 is almost never pivotal for it. It is the main strategy against Condorcet methods, STAR and BTR-score, and it targets exactly the Condorcet winner.

Diagnostic run, 60 elections, default scenario, CE with learned ballots:

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

## Axes and other outputs

| Column | Meaning |
|---|---|
| PE | proportional efficiency, sincere ballots |
| CE | Condorcet efficiency, sincere ballots |
| PE_learned, CE_learned | the same with learned ballots |
| VSE, RBE | efficiency against random candidate and random ballot, sincere ballots |
| dissonance | mean cognitive dissonance of the learned ballots (below) |
| settled | share of elections where no group changed its ballot in the last round; cycles never settle |

The front (`on_front`) uses PE, CE, PE_learned and CE_learned.

**Cognitive dissonance** (`strategy.dissonance`), per voter: for every pair the ballot orders (a above b) while the voter likes b more, add `u[b] − u[a]`; divide by the sum of `|u[a] − u[b]|` over all pairs. 0 means the ballot never claims a preference the voter does not hold (true for every sincere ballot, since rounding and leaving candidates level never reverse an order). 1 means every preference is reversed. Example, utilities 0.5 / 0.3 / 0.15 / 0.05: voting for the second choice in plurality gives 0.13, voting for the last gives 0.53.

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
| MARA | 0–5 | ratings 3–5 are approvals; order by approvals; each match adds voters preferring to approvals |
| MARA-Top | 0–5 | as MARA, but only a 5 counts as approval |
| Anti-plurality | veto (one name against) | fewest vetoes wins; sincere: against the least liked |
| Condorcet-random-drop | ranking | Condorcet winner if there is one; otherwise voters are dropped one at a time in random order until the rest have one. Averaged over 32 fixed random orders (the same on every call), so probabilities come in steps of 1/32 |
| SICT | ranking, ties allowed | Symmetrical ICT: p(X,Y) if (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top); X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins; none or all unbeaten: most first places; otherwise most first places among the unbeaten. Sincere ballots are strict; ties are learned |
| MDDA | truncated ranking (ranked = approved) | disqualify every candidate that more than half the voters rank some other candidate above (unless all would be); most approved remaining wins. Sincere: rank everyone above the mean |
| BTR-IRV | ranking | each round the two with the fewest first choices meet pairwise; the loser is eliminated |
| Benham | ranking | IRV, but a Condorcet winner among the remaining candidates wins at once |

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.

## Analysis

`analyze.py` reads one or more elections CSVs. Rows with the same parameters except the seed form one **condition**; seeds are pooled as replications.

**Uncertainty: paired bootstrap.** All methods face the same electorates, so each bootstrap replicate redraws whole elections with replacement and recomputes every method on that same draw (1000 replicates by default). This keeps comparisons paired: two methods can differ clearly even when the interval of each value on its own is wide, because elections differ much more from each other than methods differ within an election.

**What is reported per condition and method:**

| Output | Meaning |
|---|---|
| PE, CE, PE*, CE*, dissonance | point estimate and 95% interval of each value on its own |
| dominated by | methods at least as good on all four axes and better on one (point estimates), each with the share of bootstrap replicates in which that holds; the report shows at most three, the CSV all |
| paired differences (`--compare "A:B"`) | difference A − B per axis with a 95% interval; marked as a tie when the interval includes 0 |

**How to interpret:**
- The values are the result. Whether a method lies on the Pareto front is already interpretation: it depends on which axes count and on which methods are included. "Dominated by" is only a pointer to where to look.
- The axes are kept separate on purpose. Each carries its own value judgement (utilitarian or majoritarian, honest or strategic); no weighting into a single score.
- Read results **per condition first**. The strategic axes depend strongly on the group size (see Coordination).
- An ordering between two methods only counts if their paired difference is not a tie.

**Plots** (in `--out`):

| File | Shows |
|---|---|
| `arrows-<c>.png` | PE/CE plane, every method's shift from honest (open) to learned ballots (filled, labelled) in one picture |
| `shift-<c>.png` | the same shifts as one row per method, PE and CE panels; easier to read per method |
| `dissonance-<c>.png` | PE* and CE* against dissonance |
| `axes-heatmap.png` | the four axes, methods × conditions (when there are several) |

Methods are identified by labels, not colour. Random ballot and Random pair runoff are left out of the scatter plots by default (`--hide`), since their distance compresses the scale; they stay in tables and heatmaps.

Sample size: with 160 elections per condition the intervals of single values are about ±3 points and many paired differences are still ties. For conclusions, several hundred elections per condition and at least two seeds are advisable.
