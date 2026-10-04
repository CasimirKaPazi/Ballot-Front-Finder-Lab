# Voting Pareto Lab

Python 3 + NumPy only. Honest and learned-strategy elections, evaluated on a Pareto front.

```
python test_methods.py           # hand-built checks
python run.py --elections 200    # one scenario, all methods, sincere and learned ballots
```

400 elections with all 19 methods take about 3.5 minutes on 2 cores; time scales linearly with cores.

## Files

| File | Contents |
|---|---|
| `scenario.py` | Electorate generator: spatial clusters, absolute utilities `exp(-distance / width)` |
| `methods.py` | Ballot types, sincere ballots, exact tallies on ballot matrices (ties shared) |
| `strategy.py` | Learning engine and cognitive dissonance |
| `metrics.py` | Efficiencies for three baselines, Condorcet efficiency, Pareto front |
| `run.py` | Tallies every electorate with sincere and learned ballots, runs in parallel, writes CSV |
| `test_methods.py` | Hand-built checks |

## Strategy engine

- **Groups:** voters with the same sincere ranking. A group either votes sincerely (each member their own sincere ballot) or casts one shared ballot.
- **Polls:** each round draws 10 bootstrap resamples of the electorate. A proposal is judged by the group's total expected utility averaged over these samples, so a group gains in proportion to how often it would be pivotal, not only when it is decisive in the exact electorate.
- **Proposals:** one small change to the group's ballot:
  - one name: switch to another candidate
  - ranking: swap two candidates
  - approval: toggle one candidate (never empty or full)
  - ranking with approval line: swap two candidates, or move the line up or down by one
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
| `--workers` | all cores | parallel processes |

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

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.
