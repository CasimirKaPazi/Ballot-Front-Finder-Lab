# Voting Pareto Lab

Python 3 + NumPy only. Honest and learned-strategy elections, evaluated on a Pareto front.

```
python test_methods.py           # hand-built checks
python run.py --elections 200    # one scenario, all methods, sincere and learned ballots
```

400 elections take about 4 minutes on 2 cores.

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
  - score: set one candidate to a different value 0–5

  A group that deviates sometimes (15%) proposes going back to sincere.
- **Acceptance:** a proposal is kept only if it raises the group's expected utility.
- **Rounds:** 15, groups in random order each round.

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
