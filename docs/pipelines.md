# Composing methods: a proposal

Status: proposal for a version after 1.0, open for comments (2026-10-22). Nothing here is implemented. The aim is a notation that works like a small standard: clear to read, the same inline and in add-ons, and compatible with ordinary Python, so that people can build methods from parts instead of writing each one out.

## Why

Many methods are built from the same parts:

- **a narrower field:** Smith//Score is "the Smith set, then the highest score total in it", and BSR is "the Buck-set, then Score on ballots stretched over the set";
- **a measure per candidate:** score totals, winning votes, worst defeats;
- **a way to combine them:** MMWV+Score adds a score share and a defeat; MMPOWV multiplies two;
- **a last step:** the highest value, a bottom-two climb (BTR), or a real runoff.

Today each method is written by hand, so its parts cannot be reused and two methods that share a part may differ in small ways nobody intended. Composing them from named parts makes the differences visible and new combinations cheap.

## Principles

1. **One notation everywhere.** An expression typed on the command line and a definition in an add-on are the same text. The notation is a small subset of Python expressions: an add-on writes it as Python and gives it a name; the command line passes it as a string, which the lab reads with Python's own parser and accepts only the known parts (no other code runs).
2. **Names stay names.** Existing methods appear as quoted strings, `"Score"`, `"BTR-score"`, `"Smith//Score"`. Everything else is a part of the notation. A method called `A//B` is never read as a composition, and the literature's `A//B` (which often differs from the lab's A and B in detail) is not used as an operator.
3. **Parts are values, not only steps.** Some parts narrow the field (sets), some give a number per candidate (values) or per pair of candidates (pairs). Values can be added, multiplied and passed on like numbers, so custom arithmetic needs no new machinery.
4. **Compatible with ordinary functions.** A Python function that takes values and returns values can be used as a part, so an add-on can mix composed parts with its own code.
5. **Exact, with ties.** Every result is a weighted list of alternatives; a tie branches with equal weights, as the lab's methods do now. Single-winner results end in single candidates; multi-winner results would end in sets of k (committees) with the same machinery.
6. **No change to the rest of the lab.** A composed method becomes an ordinary method (ballot type, features, decision), so the strategy engine, the measures and the reports treat it like any other.

## The parts (sketch)

**Ballot readings** (per candidate or per pair, on the current field):
- `score`: score totals; `share(score)` as a share of the maximum;
- `approvals(level)`: voters rating at or above a level;
- `levels`: voters per score level (for Bucklin-like parts);
- pairwise: `margins`, `wv` (winning votes), `po` (pairwise opposition), `soft(0.4)` (soft margins, as Soft Minimax).

**Sets** (narrow the field; `|` union, `&` intersection):
- `smith`: the smallest set beating everyone outside it head to head;
- `top(k, values)`: the k best by a measure, ties branching;
- `buck(levels)`: the candidates with the most voters (or points) in the top k levels at some depth;
- `best(values)`: the highest, ties branching (a set of one, or several tied);
- later: `condorcet`, `copeland`, `approved(...)`, `above(...)`, `challenger(of, pairs)`.

**Transforms:**
- `normalize`: every ballot stretched over the current field, its lowest candidate there 0, its highest the top score. Not rounded: rounding to the score range behaves oddly and is not recommended. For ratings kept on the standard scale, the MD-type methods are the right tool.

**Reductions of pairs to values:** `worst(pairs)` (each candidate's worst result against any other; Minimax reads `worst(margins)`), later `best_against`, sums.

**Final steps:**
- `"Name"`: any existing method, run on the current field and reading;
- `best(values)` with one candidate left;
- later: `runoff` (a real, separate runoff between two finalists, on two-round ballots), `btr(order, match)` (bottom-two climb), `repeat(...)` (eliminations, as INS).

**Chaining:** `>>` reads "then": the left side narrows the field (or transforms the reading), the right side works on what is left.

## Examples (sketch; spelling to be settled)

```
smith >> "Score"                                   Smith//Score
buck(levels) >> normalize >> "Score"               BSR
buck(points) >> normalize >> "Score"               BSR-S
buck(levels) >> "BTR-score"                        a new combination
best(-worst(margins))                              Minimax
best(share(score) - worst(wv) / n)                 MMWV+Score
top(2, score) >> runoff                            Score runoff
(best(score) | challenger(best(score), margins)) >> runoff      SCR
```

SCR shows why sets need union: the Score winner and their strongest challenger are two separate results of the same field, joined before the runoff. The repeated `best(score)` is computed once.

In an add-on the same text is Python, with a name:

```python
from compose import smith, buck, levels, normalize, score, wv, worst, share, best, n

METHODS = {
    "BSR": composed(buck(levels) >> normalize >> "Score"),
    "My mix": composed(best(share(score) * (1 - worst(wv) / n))),
}
```

and a function of its own can join in:

```python
def blend(s, w):                 # values in, values out
    return s - 0.5 * w

METHODS = {"Blend": composed(best(blend(share(score), worst(wv) / n)))}
```

On the command line: `python run.py --methods 'buck(levels) >> "BTR-score"' Score STAR`. Results store the expression in a fixed canonical spelling, so the same composition always has the same name and the reports can read it back.

## How it would run

- **Fast where possible.** The lab sums per-voter features, so a poll is a weighted sum, not a recount. Most readings can be restricted to a narrower field from those sums: per-candidate features of score, grade and approval ballots by their columns; pairwise features by their submatrix; per-subset features (first choices among a set, stretched scores) by the row of the set. A composition whose readings all restrict this way compiles to an ordinary feature-based method.
- **Exact fallback.** Otherwise each part recomputes from the ballots restricted to the current field (scores as cast, rankings as the induced ranking among the field). Exact but slower, as Geometric median is now. The tests run every composition both ways and require the same result.
- **Ballot types.** Each part states the ballot types it works on; a composition has one, and a mismatch is reported with the part's name. `runoff` switches the composition to the two-round ballot.
- **Reuse of results.** A part that appears twice in one composition (the repeated `best(score)` in SCR) is computed once. Across methods, results can be shared only as far as two compositions agree from the start (the same ballots, the same parts in the same order): `smith >> "Score"` and `smith >> "BTR-score"` could share the Smith set. The gain is smaller than it looks: with learned ballots every method has its own ballots, so sharing across methods applies to the honest count only, and the time goes into the poll rounds, which each method runs on its own ballots. Within one method, results can also be shared between the steps of one count, which is where most of the repetition is.
- **Existing methods** can then be written as compositions where that is exact. Each rewrite has to reproduce the hand-written version exactly (the regression check), so this can happen method by method.

## Phases (after 1.0)

1. The core: the expression reader, sets, values and pairs with ties, `>>`, `|`, `&`, `normalize`, quoted methods as final steps, a few parts (`smith`, `top`, `buck`, `best`, `score`, `margins`, `wv`, `worst`), fast and fallback mode, named compositions in add-ons. Checked against Smith//Score, BSR, BSR-S, Minimax, MMWV+Score.
2. More parts, and `runoff` (SCR, Score runoff).
3. Orders and iteration (`btr`, `repeat`), then rewriting existing methods where exact.
4. Multi-winner results.

## Open questions

- **Spelling.** `>>` for "then", `|` and `&` for sets, plain names for readings (`score`) or calls (`score()`)? Python allows either; plain names read better, calls are more explicit.
- **Telling the kinds apart.** Readings, sets, transforms and final steps are different kinds of things, and the names could show it. Options: (a) a grammatical convention: readings are nouns for quantities (`score`, `margins`, `wv`), sets are named for the set (`smith`, `top(...)`, `buck(...)`), transforms and final steps are verbs (`normalize`, `runoff`); (b) namespaces (`read.score`, `sets.smith`, `then.normalize`), explicit but longer; (c) capitalised sets (`Smith`, `Top(...)`). In every case the reader checks the kinds and names them in its messages (`top(2, smith)`: "top needs values, smith is a set"), so a mistake is caught either way; the question is only how much the text itself should show.
- **Where values live.** Values are tied to the field they were computed on; when the field narrows, do earlier values follow (restricted) or are they recomputed? Recomputing is cleaner (a normalized reading changes them); restricting is what a voter-facing description usually means.
- **Group in the plots:** a `composed` panel, or the group of the last step.
- **Strategy:** voters see polls of the composed result as a whole, as for every method now. Worth stating so it stays a decision.
