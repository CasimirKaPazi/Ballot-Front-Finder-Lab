# Writing your own method: a tutorial

This guide is for people who want to test a voting method of their own and know little Python. It first walks through a real add-on, IDS (`addons/ids.py`), line by line. Then it builds a new one from scratch, Bucklin, the way you would build yours. The reference for the file format is README, Methods, Add-ons.

You need no Python beyond what is explained here. Copying and adapting the examples is the intended way to start.

## 0. Before you start: what is already known

Voting method design has a long history, and many ideas have been tried, named and analysed. A few minutes of reading often shows that an idea exists already, under which conditions it fails, or which small change fixes a known problem. Good places to start:

- [Techniques of method design](https://electowiki.org/wiki/Techniques_of_method_design) (Electowiki): the building blocks most methods are made of. Scores, ways to measure pairwise defeats, steps such as elimination and runoff, special sets of candidates (Smith set, Schwartz set, …), orderings and randomization, each with its known trade-offs.
- [Electowiki](https://electowiki.org/) as a whole: a page for most named methods, often with their history and known properties. Searching for the name of your idea, or a description of it, is worth trying.
- [Voting criteria](https://en.wikipedia.org/wiki/Voting_criteria) (Wikipedia): the formal properties a method can have or fail. Examples are monotonicity (more support never hurts), participation (voting never hurts the voter), independence of clones and later-no-harm. The lab measures some of these effects in practice (participation, clones with `--clones`), but knowing in advance which criteria a design fails tells you what to look for in the results.
- [Summability criterion](https://electowiki.org/wiki/Summability_criterion): see Section 1.

The lab then adds what criteria cannot say: how often a weakness matters in realistic electorates, and what happens when voters respond to it.

## 1. What the simulator needs from a method

A method has two parts:

1. **A ballot type**, chosen from the built-in ones: `"rank"`, `"score"`, `"approval"` and so on (list below). The ballot type tells the simulator how honest voters fill in a ballot, how lazy voters cut it short, which strategic changes they can try, and how sincerity and fidelity are measured. You don't write any of that; you pick the type. (All of it is defined per type in `ballots.py`; a genuinely new ballot format can be added there or in an add-on, but that is rarely needed and not covered here.)
2. **A counting rule**: given everyone's ballots, who wins.

The counting rule is split in two, for speed. During strategy every method is tallied thousands of times, on many poll samples. So a method doesn't look at all ballots at once. Instead:

- A **feature** turns *one* ballot into some numbers: "this voter ranks A above B", or "this voter's first choice is C".
- The simulator **adds up** each feature over all voters (and over the voters of each poll sample). Adding is very fast.
- The **decision** looks only at these sums and returns who wins.

This is the idea of **precinct summability** ([summability criterion](https://electowiki.org/wiki/Summability_criterion); on Wikipedia: [batch summability](https://en.wikipedia.org/wiki/Batch_summability)). A method is summable if every precinct can count its ballots into a small summary, and the summaries can simply be added up:
- Plurality, approval, score and Borda need one number per candidate.
- Condorcet methods need one number per pair of candidates (who is preferred to whom, by how many).
- IRV is not summable in this sense. Which candidate a ballot counts for depends on who has been eliminated, so the summary needs a count for every possible set of remaining candidates, which grows exponentially. The lab does exactly that (the `"firsts"` feature, 2^C sets): fine for 6 candidates, slow for 15.

Whether a rule is summable is worth knowing for its own sake, since it matters for counting real elections, and the feature you write tells you right away. If your rule can't be written this way at all (it needs to see individual ballots), there is a slower way out, explained in Section 5.

### The ballot matrix

All ballots of an election form a table `B` with one row per voter and one column per candidate. The candidates are numbered 0, 1, 2, … (A = 0, B = 1, C = 2). What the numbers in a row mean depends on the ballot type:

| Ballot type | One voter's row means | Example with 3 candidates |
|---|---|---|
| `"score"` | the score of each candidate, 0 to `methods.MAXS` (default 5) | `[5, 3, 0]`: A 5, B 3, C 0 |
| `"rank"` | rank points: the first gets C−1, the next C−2, …; unranked candidates (truncated ballot) get 0, tied | `[2, 1, 0]`: A > B > C; `[0, 2, 0]`: only B ranked |
| `"approval"` | 1 approved, 0 not | `[1, 1, 0]` |
| `"fav"` | 1 for the one candidate named | `[0, 1, 0]` |
| `"weak"` | levels, equal = tied (rankings with ties) | `[2, 2, 0]`: A = B > C |
| `"grade3"` | 0, 1, 2 (bad, OK, good) | `[2, 0, 1]` |

The others (`"fav_run"`, `"app_run"`, `"veto"`, `"trunc"`, `"trunc3"`, `"strict"`) are described at the top of `methods.py`.

## 2. Five pieces of Python you'll meet

- **Variables and functions.** `def name(x):` starts a function. Its body is the indented lines below it. `return` gives back the result.
- **Comments.** Everything after `#` on a line is a comment for humans. Text in triple quotes `"""…"""` at the start of a file or function is its description.
- **Dictionaries**, written `{key: value, …}`, look things up by name. `METHODS = {"IDS": Method(…)}` means "the method called IDS is this".
- **NumPy arrays** (`np.…`) are the tables of numbers. `B.shape` is their size, e.g. `(99, 6)` for 99 voters and 6 candidates. Arithmetic works on whole tables at once: `B * 2` doubles every entry, and `B > 0` gives a table of True/False.
- **Indexing.** `x[i]` is entry i, counting from 0. `x[:, c]` is column c. `x[mask]` picks the entries where `mask` is True.

To check that something does what you think, you can type `python3` in the project folder and try it line by line:

```
>>> import numpy as np
>>> B = np.array([[5, 3, 0], [0, 5, 3]])
>>> B.sum(axis=0)        # add up the rows: one total per candidate
array([5, 8, 3])
```

## 3. Walkthrough: IDS (`addons/ids.py`)

IDS (iterated dropped score) works like this. Each round, on every ballot the lowest-rated remaining candidates are struck out, so they count 0, and the other scores stay. The candidate with the lowest total is eliminated. This repeats until one is left.

### The feature: scores with the lowest struck out, for every possible set of remaining candidates

The elimination needs the totals over *the candidates still in the race*. That set changes every round, and the decision only sees sums, never ballots. So the feature prepares the totals for **every possible set** of remaining candidates in advance. With 6 candidates there are 2⁶ = 64 sets, which is still small.

How are sets numbered? Each candidate is one bit of a number: A = 1, B = 2, C = 4, D = 8, … The set {A, C} is 1 + 4 = 5, and the set of all three, {A, B, C}, is 1 + 2 + 4 = 7. `methods.masks(C)` gives a table with one row per set number, True where the candidate is in that set.

```python
def dscore_sub(B):
    n, C = B.shape
    m = masks(C)[None]
    lo = np.where(m, B[:, None, :], np.inf).min(axis=2, keepdims=True)
    return np.where(m & (B[:, None, :] > lo), B[:, None, :], 0.0).reshape(n, -1)
```

Line by line:

- `n, C = B.shape`: n voters, C candidates.
- `m = masks(C)[None]`: the table of sets. `[None]` adds an extra dimension so that it lines up with every voter.
- `lo = …`: for every voter and every set, the lowest score *among the candidates in the set*. `np.where(m, B…, np.inf)` replaces the scores of candidates outside the set by infinity, so `.min(…)` ignores them.
- `return np.where(m & (B… > lo), B…, 0.0)`: keep a score only if the candidate is in the set **and** scored above that lowest score. Everything else becomes 0: the lowest ones (struck out) and the candidates outside the set. If a voter gives everyone in the set the same score, nobody is above the lowest, so the whole ballot counts 0 for that set.
- `.reshape(n, -1)`: flatten each voter's 64 × C table into one long row. The simulator wants one row of numbers per voter.

For one voter with `A5 B4 C0` and the set {A, B} (number 3), the lowest in the set is B's 4. So the row for set 3 becomes `A5 B0 C0`. For the set of all three (number 7), the lowest is C, so it becomes `A5 B4 C0`.

After the simulator has added this feature up over all voters, `s["dscore_sub"][set number]` is the table of totals for that set.

### The decision: eliminate the lowest, round by round

```python
def decide(s):
    return methods.d_iterated_normalized_score(s, "dscore_sub")
```

IDS reuses the elimination that INS already has in `methods.py`, only on the dropped scores instead of INS's stretched ones. That function looks like this:

```python
def d_iterated_normalized_score(s, key="nscore_sub"):
    sub = s[key]                      # the summed feature: totals for every set
    C = sub.shape[1]                  # number of candidates
    P = np.zeros(C)                   # winning chances, filled in below

    def go(alive, w):                 # alive: candidates still in; w: chance of reaching this point
        if len(alive) == 1:           # one left: it wins (with chance w)
            P[alive[0]] += w
            return
        pts = sub[int(sum(1 << c for c in alive))][alive]   # totals over the set "alive"
        low = np.flatnonzero(pts <= pts.min() + EPS)        # the lowest (several if tied)
        for i in low:                                       # each tied lowest is out with equal chance
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)           # start with everyone in, chance 1
    return P
```

- **Set number:** `1 << c` is 2 to the power c, so `sum(1 << c for c in alive)` is the set number described above.
- **Recursion:** `go` calls itself with one candidate fewer each time. That is the round-by-round elimination.
- **Ties:** if two candidates tie for last, the function follows both possibilities, each with half the chance. The result `P` then holds exact winning chances, e.g. `[0.5, 0.5, 0]`. Every method in the lab handles ties this way, so yours should too.
- **`EPS`:** a tiny number, 0.000000001. Computers store fractions with tiny rounding errors, so "equal" is written as "within EPS".

### Registering the method

```python
FEATURES = {"dscore_sub": (dscore_sub, lambda C: (2 ** C, C))}
METHODS = {"IDS": Method("score", decide, ("dscore_sub",))}
GROUP = "rated"
```

- `FEATURES` announces the new feature. It gives the function, and the shape of its sum for C candidates: 2^C sets × C candidates. `lambda C: …` is a short way to write a one-line function.
- `METHODS` gives the method its name and says: score ballots, this decision, and it needs the feature `dscore_sub`. The comma in `("dscore_sub",)` is needed when the list has a single name.
- `GROUP` puts IDS into the existing group `rated`. Leave it out to use the group `addons`.

### The test

```python
def test():
    B = np.array([[5, 4, 0]] * 2 + [[5, 5, 1]] + [[2, 3, 3]] * 3, dtype=float)
    assert np.allclose(METHODS["IDS"].tally(B), [1, 0, 0])
```

The test builds a small election whose result you have worked out by hand (written in the file's comment): 2 voters `A5 B4 C0`, 1 voter `A5 B5 C1`, 3 voters `A2 B3 C3`. It checks that IDS elects A. `assert` stops with an error if the check fails, and `python3 test_methods.py` runs every add-on's test. A hand-checked example like this is the best protection against a counting rule that does something other than you meant.

## 4. Your own method, step by step: Bucklin

Bucklin counts first choices. If someone has a majority, the one with the most votes wins. Otherwise everyone's second choices are added, then third choices, and so on. On truncated ballots, unranked candidates are never counted.

**Step 1: copy the template.** Copy `addons/_template.py` to `addons/bucklin.py`. A file name starting with `_` is ignored, so the copy must not start with one.

**Step 2: the feature.** Bucklin needs to know, for every candidate, how many voters put it in first place, second place, and so on. Per voter, that is a C × C table with a 1 where candidate c stands at place k:

```python
def positions(B):
    n, C = B.shape
    place = (C - 1 - B).astype(int)                        # rank points C-1, C-2, ... -> places 0, 1, ...
    zero = B == 0
    unranked = zero & (zero.sum(axis=1, keepdims=True) > 1)    # 0 shared by two or more: unranked
    out = np.zeros((n, C, C))
    voters, cands = np.nonzero(~unranked)                  # every (voter, candidate) that is ranked
    out[voters, cands, place[voters, cands]] = 1
    return out.reshape(n, -1)
```

- **Places:** rank points C−1, C−2, … become places 0, 1, … (0 = first).
- **Unranked candidates:** on a truncated ranking they share the value 0. A single 0 is just the last place of a full ranking. So only a 0 that two or more candidates share means "unranked", and those candidates get no place.
- **`~`** means "not", so `~unranked` marks the ranked candidates.

**Step 3: the decision.** After summing, `s["positions"][c, k]` is the number of voters with candidate c at place k, and `s["n"]` is the number of voters (a built-in feature):

```python
def decide(s):
    table = s["positions"]
    n = s["n"]
    C = table.shape[0]
    votes = np.cumsum(table, axis=1)            # votes[c, k] = voters with c at places 0..k
    for k in range(C):                          # round 1, 2, ...
        this_round = votes[:, k]
        if this_round.max() > n / 2 or k == C - 1:
            best = np.flatnonzero(this_round >= this_round.max() - methods.EPS)
            P = np.zeros(C)
            P[best] = 1 / len(best)             # ties split the win
            return P
```

`np.cumsum(…, axis=1)` adds the places up: first, first + second, and so on. Those are exactly the Bucklin rounds.

**Step 4: register it.**

```python
FEATURES = {"positions": (positions, lambda C: (C, C))}
METHODS = {"Bucklin": Method("rank", decide, ("n", "positions"))}
GROUP = "plurality-based"
```

**Step 5: a test with a result you know.** Use 2 voters A > B > C, 2 voters C > B > A and 1 voter B > A > C. In round 1, A has 2, B 1 and C 2, and nobody has more than half of 5. In round 2, B has 1 + 2 + 2 = 5, so B wins.

```python
def test():
    B = np.array([[2, 1, 0]] * 2 + [[0, 1, 2]] * 2 + [[1, 2, 0]], dtype=float)
    assert np.allclose(METHODS["Bucklin"].tally(B), [0, 1, 0])
```

**Step 6: run it.**

```
python3 test_methods.py                                   # all checks, including "add-on bucklin.py ok"
python3 run.py --methods Bucklin IRV Schulze --elections 50 --out bucklin-test
python3 analyze.py bucklin-test
```

To compare with a bigger run you already have, run only your method with the same parameters and seed, and analyse both folders together (README, Combining runs):

```
python3 run.py --methods Bucklin --elections 600 --out bucklin
python3 analyze.py run-20261012-191423 bucklin --out compare
```

If it turns out uninteresting, delete `addons/bucklin.py`. To share it, send the file.

## 5. Checklist and pitfalls

- **Ties must split the win.** Return chances like `[0.5, 0.5, 0]`, never pick one tied candidate. The simulator relies on exact results; a hidden tie-break by candidate number would favour candidate A.
- **Compare with `EPS`:** use `x >= best - methods.EPS` rather than `x == best` for numbers that come from arithmetic.
- **Score range:** read `methods.MAXS` *inside* your functions, not at the top of the file. The range can change with `--max-score` after your file is loaded.
- **A fixed score range** (rarely wanted): `Method("score", ..., settings={"max_score": 10})` makes the method always use 0..10, whatever `--max-score` says. Name it `Name-10`, and keep a base version without fixed range, so it stays comparable with the other methods (README, Score range).
- **Features must add up.** The simulator only gives you sums over voters. Totals, counts and pairwise counts add up; a median doesn't. For a median, count how many voters gave each score (the built-in feature `"grades"`) and find the median from those counts, as MJ does.
- **Many sets are slow.** A feature over all sets of candidates (2^C rows, like IDS) is fine for 6 candidates, but doubles in size with every extra candidate.
- **If the rule can't be written with sums** (it needs individual ballots, e.g. a random ballot drawn), use `Method("rank", full=tally)` with `def tally(B): …` returning the winning chances directly from the whole ballot matrix. It works, but strategic polls become much slower.
- **Names:** method and feature names must be new. Loading stops with a message if one already exists.
- **Truncated rankings:** on `"rank"` and `"weak"` ballots, a 0 shared by two or more candidates means unranked. Decide what your rule does with them (Bucklin: never counted). README, Truncation and laziness, lists what the built-in methods do.

### Built-in features you can use without writing your own

| Name | Summed over voters, it gives | Shape |
|---|---|---|
| `"n"` | the number of voters | a single number |
| `"sum"` | the total of each candidate's ballot values (score totals; Borda points on rankings) | C |
| `"gt"` | `gt[a, b]` = voters putting a above b (pairwise counts; margins = `gt - gt.T`) | C × C |
| `"first"` | voters whose first choice is each candidate | C |
| `"last"` | voters whose last choice is each candidate (ties at the bottom shared) | C |
| `"firsts"` | for every set of remaining candidates, the first choices among them (exhausted ballots count for nobody) | 2^C × C |
| `"pos"` | voters giving each candidate more than 0 (approvals, ranked candidates) | C |
| `"grades"` | voters giving each candidate each score 0..MAXS (for medians) | C × (MAXS + 1) |
| `"top"` | voters rating each candidate at their highest level | C |

The full list, with the rest, is `FEATURES` in `methods.py`. Helper functions such as `methods.margins_of(s)` (pairwise margins from `"gt"`), `methods.d_irv` and `methods.d_iterated_normalized_score` can be reused as IDS does.
