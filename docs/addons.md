# Writing your own method: a tutorial

This guide is for people who want to test a voting method of their own and know little Python. It first walks through a real add-on, IDS (`addons/ids.py`), line by line. Then you change it into two variants yourself (Section 4, with solutions to compare against), and finally build a new method from scratch, Bucklin (Section 5). The reference for the file format is [methods.md](methods.md), Add-ons.

You will need some Python. The guide explains every piece the examples use, but writing a new counting rule is still programming: expect to try things, read error messages and test small cases by hand. Copying an example that is close to your idea and changing it step by step is the easiest way in. Section 6 lists what to do when you are stuck.

The commands below use `python`; on some systems it is called `python3`.

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

Whether a rule is summable is worth knowing for its own sake, since it matters for counting real elections, and the feature you write tells you right away. If your rule can't be written this way at all (it needs to see individual ballots), there is a slower way out, explained in Section 6.

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

The others (`"fav_run"`, `"app_run"`, `"score_run"`, `"veto"`, `"trunc"`, `"trunc3"`, `"strict"`) are described at the top of `ballots.py`.

## 2. The pieces of Python you'll meet

**Basics.**

- `def name(x):` starts a function called `name` that takes `x`. Its body is the indented lines below it. `return` hands back the result.
- Everything after `#` on a line is a comment for humans. Text in triple quotes `"""…"""` at the start of a file or function describes it.
- `{key: value, …}` is a dictionary: it looks things up by name. `METHODS = {"IDS": Method(…)}` means "the method called IDS is this".
- `x = …` stores a result under the name `x`, so the next lines can use it.

**NumPy arrays** (`np.…`) are tables of numbers, and almost everything here works on them. They can have more than two dimensions: a table of tables. `B.shape` gives the size of each dimension, e.g. `(99, 6)` for 99 voters and 6 candidates.

- **Arithmetic works on whole tables at once.** `B * 2` doubles every entry. `B > 0` compares every entry and gives a table of True/False of the same size.
- **Indexing.** `x[i]` is entry i, counting from 0. `x[:, c]` is column c (`:` means "all"). `x[mask]` picks the entries where `mask` is True.
- **Combining True/False tables:** `a & b` is True where both are, `a | b` where either is, `~a` where `a` is not.
- **`np.where(condition, x, y)`** builds a new table: it takes the entry from `x` where `condition` is True and from `y` where it is False. It is the "if … else" for whole tables.
- **`.min(axis=k)`, `.max(axis=k)`, `.sum(axis=k)`** collapse dimension k: `B.sum(axis=0)` adds up the rows (one total per candidate), `B.sum(axis=1)` adds up each row (one total per voter). With `keepdims=True` the collapsed dimension stays, with size 1, so the result still lines up with the original table.
- **`.reshape(…)`** reads the same numbers into a different shape. `.reshape(n, -1)` makes n rows and lets NumPy work out the length of each row.

**Lining up tables of different shapes ("broadcasting").** This is the least obvious piece, and the IDS code relies on it. When two tables are combined, NumPy compares their shapes dimension by dimension from the right. Where one of them has size 1, that entry is repeated to match the other. So a table of shape `(n, 1, C)` (each voter's scores, once) combined with one of shape `(1, 64, C)` (64 sets, the same for every voter) gives `(n, 64, C)`: every voter's scores, repeated for every set. `[None]` and `[:, None, :]` insert such a size-1 dimension:

```
>>> import numpy as np
>>> B = np.array([[5, 4, 0], [2, 3, 3]])      # 2 voters, 3 candidates
>>> B.shape
(2, 3)
>>> B[:, None, :].shape                         # one extra dimension in the middle
(2, 1, 3)
```

**Trying things out.** Type `python` in the project folder and enter lines one at a time; the result is printed right away. This is the quickest way to see what a line does:

```
>>> B.sum(axis=0)          # add up the rows: one total per candidate
array([7, 7, 3])
>>> B.max(axis=1)          # each voter's highest score
array([5, 3])
>>> np.where(B > 2, B, 0)  # keep scores above 2, set the rest to 0
array([[5, 4, 0],
       [0, 3, 3]])
```

## 3. Walkthrough: IDS (`addons/ids.py`)

IDS (iterated dropped score) works like this. In each round, every ballot's lowest-rated remaining candidates are struck out (they count 0), and the other scores stay as they are. The candidate with the lowest total is eliminated. This repeats until one candidate is left.

### The feature: scores with the lowest struck out, for every possible set of remaining candidates

The elimination needs each candidate's total over *the candidates still in the race*. That set changes every round, and the decision only sees sums, never single ballots. So the feature prepares the totals for **every possible set** of remaining candidates in advance. With 6 candidates there are 2⁶ = 64 sets, which is still small.

How are the sets numbered? Each candidate is one bit of a number: A = 1, B = 2, C = 4, D = 8, … The set {A, C} is 1 + 4 = 5, and the set of all three, {A, B, C}, is 1 + 2 + 4 = 7. `methods.masks(C)` returns a table with one row per set number, True where the candidate is in that set. For 3 candidates (shown as 1 and 0):

```
set 0  {}         0 0 0
set 1  {A}        1 0 0
set 2  {B}        0 1 0
set 3  {A, B}     1 1 0
set 4  {C}        0 0 1
set 5  {A, C}     1 0 1
set 6  {B, C}     0 1 1
set 7  {A, B, C}  1 1 1
```

The code:

```python
def dscore_sub(B):
    n, C = B.shape
    in_set = masks(C)[None]                    # (1, 2^C, C): True where a candidate is in the set
    scores = B[:, None, :]                     # (n, 1, C): each voter's scores, the same for every set
    lowest = np.where(in_set, scores, np.inf).min(axis=2, keepdims=True)   # (n, 2^C, 1)
    keep = in_set & (scores > lowest)          # in the set and above the set's lowest score
    return np.where(keep, scores, 0.0).reshape(n, -1)
```

Line by line, with one voter who scores `A5 B4 C0`:

1. `n, C = B.shape` reads the number of voters and of candidates from the ballot table.
2. `in_set = masks(C)[None]` takes the table of sets above. `[None]` puts a size-1 dimension in front, so that the sets line up with every voter (Section 2, broadcasting).
3. `scores = B[:, None, :]` adds a size-1 dimension in the middle of the ballots, so that every voter's scores line up with every set.
4. `np.where(in_set, scores, np.inf)` writes each voter's scores into every set, but puts infinity (`np.inf`) wherever a candidate is *not* in the set. For our voter:
   ```
   set 3  {A, B}     5  4  inf
   set 5  {A, C}     5  inf  0
   set 7  {A, B, C}  5  4  0
   ```
   `.min(axis=2, keepdims=True)` then finds the smallest number in each row. Since infinity is never the smallest, this is **the lowest score among the candidates in the set**: 4 for set 3, 0 for sets 5 and 7. The result is stored as `lowest`.
5. `keep = in_set & (scores > lowest)` marks the candidates that are in the set **and** scored above that lowest score. In set 3 only A is kept; in set 7, A and B.
6. `np.where(keep, scores, 0.0)` keeps those scores and sets everything else to 0: the struck-out lowest candidates and those outside the set. For our voter, set 3 becomes `A5 B0 C0` and set 7 stays `A5 B4 C0`. If a voter gives everyone in a set the same score, nobody is above the lowest, so the ballot counts 0 for that set.
7. `.reshape(n, -1)` flattens each voter's 8 × 3 table (64 × 6 with six candidates) into one long row. The simulator wants exactly one row of numbers per voter.

After the simulator has added this feature up over all voters, `s["dscore_sub"][set number]` holds the candidates' totals for that set.

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

- **Set number:** `1 << c` is 2 to the power c, so `sum(1 << c for c in alive)` computes the set number described above.
- **Recursion:** `go` calls itself with one candidate fewer each time. That is the round-by-round elimination.
- **Ties:** if two candidates tie for last, the function follows both possibilities, each with half the chance. `P` then holds exact winning chances, e.g. `[0.5, 0.5, 0]`. Every method in the lab handles ties this way, so yours should too.
- **`EPS`:** a tiny number, 0.000000001. Computers store fractions with tiny rounding errors, so "equal" is written as "within EPS".

Because the elimination only reads `s[key]`, **any feature with the same layout (one row of totals per set) can reuse it.** That makes variants of IDS and INS easy: write a new feature, keep the decision.

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

The test builds a small election whose result was worked out by hand (written in the file's comment): 2 voters `A5 B4 C0`, 1 voter `A5 B5 C1`, 3 voters `A2 B3 C3`. `[[5, 4, 0]] * 2` repeats a row twice, and `+` joins the lists of rows. The test checks that IDS elects A: `assert` stops with an error if the check fails, and `python test_methods.py` runs every add-on's test. A hand-checked example like this is the best protection against a counting rule that does something other than you meant.

## 4. Exercise: two variants of IDS

IDS lowers the bottom of each ballot. Two natural variants:

- **IMS (iterated maxed score):** instead of striking the lowest, *raise* each ballot's highest-rated remaining candidates to the top score (`methods.MAXS`). The other scores stay.
- **IDMS (both):** raise the highest to the top score *and* strike the lowest; the scores in between stay. (INS goes one step further and stretches the ones in between too.)

The names are placeholders; choose your own. Only the feature changes: the decision and the registration work as for IDS. Try it yourself before opening the solutions.

**Steps.**

1. Copy `addons/ids.py` to a new file, e.g. `addons/ids_variants.py`. Rename the feature function (e.g. `mscore_sub`), since feature names must be new.
2. In the feature, you need **the highest score among the candidates in the set**, the mirror image of `lowest`. Hint: outside the set, put a value that is never the highest, and take `.max` instead of `.min`.
3. Mark the candidates to raise: in the set **and** at that highest score.
4. Build the result with `np.where` twice: where a candidate is to be raised, `methods.MAXS`; otherwise, its score if it is in the set, else 0. (An `np.where` can be the "else" part of another `np.where`.)
5. For IDMS: compute both `lowest` and `highest`, and decide what a ballot that rates everyone in the set alike should do. IDS lets it count for nobody, and that is a sensible choice here too: then `highest > lowest` must hold for a candidate to be raised.
6. Register both methods. One file can hold several features and methods: put them all into the same `FEATURES` and `METHODS` dictionaries. Since the decision only differs in the feature name, a `lambda` is enough: `lambda s: methods.d_iterated_normalized_score(s, "mscore_sub")`.
7. Work out a small election by hand and write it as a test (Section 3). The IDS example works: with the default range 0–5, both variants elect B there. (By hand for IMS: in {A, B, C} the third group's ballot `A2 B3 C3` becomes `A2 B5 C5`, giving A 21, B 28, C 16; C is out. In {A, B} it becomes `A2 B5`, giving A 21, B 28: B wins.)
8. Run `python test_methods.py`, then a small run: `python run.py --methods IDS IMS IDMS INS --elections 50 --out ids-variants`.

To look at what your feature does for one ballot, try it in `python` (from the project folder):

```
>>> import numpy as np, methods
>>> import importlib.util as u; spec = u.spec_from_file_location("v", "addons/ids_variants.py")
>>> v = u.module_from_spec(spec); spec.loader.exec_module(v)
>>> v.mscore_sub(np.array([[3., 2, 0]])).reshape(8, 3)[[3, 7]]     # sets {A, B} and {A, B, C}
array([[5., 2., 0.],
       [5., 2., 0.]])
```

(Loading the file this way runs it a second time, separately from the simulator; that is fine for trying things out.)

<details>
<summary>Solution: IMS</summary>

```python
def mscore_sub(B):
    """For every subset S: the voter's highest-rated candidates in S raised to the top score
    (methods.MAXS), the other scores in S unchanged; 0 outside S."""
    n, C = B.shape
    in_set = masks(C)[None]
    scores = B[:, None, :]
    highest = np.where(in_set, scores, -np.inf).max(axis=2, keepdims=True)
    top = in_set & (scores >= highest)
    inside = np.where(in_set, scores, 0.0)            # the scores in the set, 0 outside
    return np.where(top, methods.MAXS, inside).reshape(n, -1)
```

`-np.inf` (minus infinity) is never the highest, just as infinity is never the lowest.
</details>

<details>
<summary>Solution: IDMS</summary>

```python
def dmscore_sub(B):
    """For every subset S: the highest-rated candidates in S raised to the top score, the lowest
    struck (0), the others unchanged; 0 outside S. A ballot rating everyone in S alike counts for none."""
    n, C = B.shape
    in_set = masks(C)[None]
    scores = B[:, None, :]
    lowest = np.where(in_set, scores, np.inf).min(axis=2, keepdims=True)
    highest = np.where(in_set, scores, -np.inf).max(axis=2, keepdims=True)
    top = in_set & (scores >= highest) & (highest > lowest)
    above_lowest = in_set & (scores > lowest)
    return np.where(top, methods.MAXS, np.where(above_lowest, scores, 0.0)).reshape(n, -1)
```

The registration for both:

```python
FEATURES = {"mscore_sub": (mscore_sub, lambda C: (2 ** C, C)),
            "dmscore_sub": (dmscore_sub, lambda C: (2 ** C, C))}
METHODS = {
    "IMS": Method("score", lambda s: methods.d_iterated_normalized_score(s, "mscore_sub"), ("mscore_sub",)),
    "IDMS": Method("score", lambda s: methods.d_iterated_normalized_score(s, "dmscore_sub"), ("dmscore_sub",)),
}
GROUP = "rated"


def test():
    B = np.array([[5, 4, 0]] * 2 + [[5, 5, 1]] + [[2, 3, 3]] * 3, dtype=float)
    assert np.allclose(METHODS["IMS"].tally(B), [0, 1, 0])
    assert np.allclose(METHODS["IDMS"].tally(B), [0, 1, 0])
```

Keep the `import` lines at the top of the copied file (`import numpy as np`, `import methods`, `from methods import Method, masks`). The file must not define `"IDS"` or `dscore_sub` again, since those names exist already (`ids.py` is loaded too).
</details>

## 5. Your own method, step by step: Bucklin

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
python test_methods.py                                   # all checks, including "add-on bucklin.py ok"
python run.py --methods Bucklin IRV Schulze --elections 50 --out bucklin-test
python analyze.py bucklin-test
```

To compare with a bigger run you already have, run only your method with the same parameters and seed, and analyse both folders together ([running.md](running.md), Combining runs):

```
python run.py --methods Bucklin --elections 600 --out bucklin
python analyze.py run-20261012-191423 bucklin --out compare
```

If it turns out uninteresting, delete `addons/bucklin.py`. To share it, send the file.

## 6. Checklist, pitfalls and getting unstuck

- **Ties must split the win.** Return chances like `[0.5, 0.5, 0]`, never pick one tied candidate. The simulator relies on exact results; a hidden tie-break by candidate number would favour candidate A.
- **Compare with `EPS`:** use `x >= best - methods.EPS` rather than `x == best` for numbers that come from arithmetic.
- **Score range:** read `methods.MAXS` *inside* your functions, not at the top of the file. The range can change with `--max-score` after your file is loaded.
- **A fixed score range** (rarely wanted): `Method("score", ..., settings={"max_score": 10})` makes the method always use 0..10, whatever `--max-score` says. Name it `Name-10`, and keep a base version without fixed range, so it stays comparable with the other methods ([model.md](model.md), Score range).
- **Features must add up.** The simulator only gives you sums over voters. Totals, counts and pairwise counts add up; a median doesn't. For a median, count how many voters gave each score (the built-in feature `"grades"`) and find the median from those counts, as MJ does.
- **Many sets are slow.** A feature over all sets of candidates (2^C rows, like IDS) is fine for 6 candidates, but doubles in size with every extra candidate.
- **If the rule can't be written with sums** (it needs individual ballots, e.g. a random ballot drawn), use `Method("rank", full=tally)` with `def tally(B): …` returning the winning chances directly from the whole ballot matrix. It works, but strategic polls become much slower.
- **Names:** method and feature names must be new. Loading stops with a message if one already exists.
- **Truncated rankings:** on `"rank"` and `"weak"` ballots, a 0 shared by two or more candidates means unranked. Decide what your rule does with them (Bucklin: never counted). [model.md](model.md), Truncation and laziness, lists what the built-in methods do.

**When you are stuck:**

- **Read the error message from the bottom up.** The last line says what went wrong (e.g. `NameError: name 'x' is not defined`: a typo or a missing line); the lines above it give the file and line number.
- **Shrink the problem.** Try the feature on a single ballot with 3 candidates in `python` (as in Section 4), and compare every intermediate table with what you expect by hand. `print(x.shape)` or `print(x)` inside a function shows what it holds while the tests run.
- **Shapes cause most errors.** If NumPy complains that shapes "could not be broadcast", write the shape of each table next to its line as a comment, as `ids.py` does, and check that the dimensions line up (Section 2).
- **Start from the closest example.** IDS for anything over sets of candidates, Bucklin for anything over ranking places, the template for a plain total.
- **Ask for help with the file and the error.** Sharing the add-on file, a small ballot example and the full error message makes it easy for someone else (or an AI assistant) to see the problem.

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
