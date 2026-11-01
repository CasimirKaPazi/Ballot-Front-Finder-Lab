# Composing methods: a plan

Status: phases 1 and 2 built (2026-11-01; compose.py, described for users in [methods.md](methods.md), Compositions); phases 3 and 4 are plans. First proposal 2026-10-22, revised 2026-11-01. The aim is a notation that works like a small standard: clear to read, the same inline and in add-ons, and compatible with ordinary Python, so that people can build methods from parts instead of writing each one out.

## Why

Many methods are built from the same parts:

- **a narrower field:** Smith//Score is "the Smith set, then the highest score total in it", and ESR is "the candidates ahead in score points at some depth, then Score on ballots stretched over that set";
- **a measure per candidate:** score totals, winning votes, worst defeats;
- **a way to combine them:** MMWV+Score adds a score share and a defeat; MMWVxScore multiplies two;
- **a last step:** the highest value, a bottom-two climb (BTR), or a real runoff.

Today each method is written by hand, so its parts cannot be reused and two methods that share a part may differ in small ways nobody intended. Composing them from named parts makes the differences visible and new combinations cheap.

## Principles

1. **One notation everywhere.** An expression typed on the command line and a definition in an add-on are the same text. The notation is a small subset of Python expressions: an add-on writes it as Python and gives it a name; the command line passes it as a string, which the lab reads with Python's own parser and accepts only the known parts (no other code runs).
2. **Names stay names.** Existing methods appear as quoted strings, `"Score"`, `"BTR-score"`, `"Smith//Score"`. Everything else is a part of the notation. A method called `A//B` is never read as a composition, and the literature's `A//B` (which often differs from the lab's A and B in detail) is not used as an operator.
3. **Parts are values, not only steps.** Some parts narrow the field (sets), some give a number per candidate (values) or per pair of candidates (pairs). Values can be added, multiplied and passed on like numbers, so custom arithmetic needs no new machinery.
4. **Compatible with ordinary functions.** A Python function that takes values and returns values can be used as a part, so an add-on can mix composed parts with its own code.
5. **Exact, with ties.** Every result is a weighted list of alternatives; a tie branches with equal weights, as the lab's methods do now. Single-winner results end in single candidates; multi-winner results would end in sets of k (committees) with the same machinery.
6. **No change to the rest of the lab.** A composed method becomes an ordinary method (`Method`: ballot type, features, decision, and its poll view and settings), so both strategy engines, the measures and the reports treat it like any other.

## The state: field and readings, passed through

A composition is a chain of steps over one state: the **field** (the candidates still in) and the **readings** known so far (score totals, the pairwise matrix, ...). Each step changes only what it is about:

- **a set** (`smith`, `top`, `best`, ...) narrows the field and passes every reading through unchanged;
- **a transform** (`normalize`) replaces a reading by a new one; later steps see the new one;
- **a reading or a final step** takes the latest version of what it needs, restricted to the current field: if the step before did not produce it, it falls back to the step before that, as far back as the ballots.

So `smith >> "Score"` is: ballots (score) → `smith` (narrows the field, scores untouched) → `"Score"` (the score totals of the ballots, on the Smith set only), which is Smith//Score as written by hand. Values are restricted and passed through, never silently recomputed.

**Readings that depend on the field.** Passing through is exact for a reading that does not depend on which other candidates are in: a candidate's score total, a pair's head-to-head count; narrowing just drops columns or rows. Some readings do depend on the field: first preferences (whose favourite among those left?), last places, rank points (Borda), scores stretched over the field. Falling back to an earlier field would change their meaning without anyone noticing: `smith >> "Plurality"` would count favourites among all candidates, not among the Smith set. So each reading states whether it depends on the field:

- **independent of the field:** passed through and restricted;
- **dependent on the field:** computed again on the current field: from totals over every subset of candidates (exact, but not what a poll shows; the poll then uses the stand-in, see Polls), or by the exact fallback from the ballots restricted to the field (How it would run). Only for a ballot type that cannot be restricted (an add-on's own type) does the reader report an error naming the reading. It never falls back to an earlier field.

The features as they are today (methods.py, `FEATURES`):

| Kind | Features | On a narrower field |
|---|---|---|
| per candidate, independent | `sum` (score, grade and approval ballots), `app`, `run_sum`, `maxs`, `pos`, `upper`, `ge1`, `ge2`, `eq0`, `levels`, `grades` | the columns of the field |
| per pair, independent | `gt`, `runoff_gt`, `excess`, `soft20`, `soft40`, `levelled`, `diffhist` | the rows and columns of the field |
| per subset, among it | `firsts`, `firsts_eq`, `firsts_ex`, `lasts`, `keepnorm`, `nscore_sub` | the entries of the field's subsets (exact; not poll-like) |
| dependent on the field | `first`, `last`, `top`, `eqtop`, `eqbot`, `first_split`, `transfer`, `transfer_whole`, `lasttransfer`, `favscore`, `dowdall`, `firstsecond`; `fset`, `fsetB`, `solid`, `acquiesce` (per subset, but judged against everyone outside it); `sum` on anything but rated ballots (rank points, marks) | from a subset feature if there is one, else the exact fallback (phase 2), else an error |
| always | `n` | unchanged |

This table also decides which quoted methods can be a final step on a narrower field from totals alone: those whose features are all independent of the field, or have a subset form.

## The parts (sketch)

**Ballot readings** (per candidate or per pair):
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
- `"Name"`: any existing method, run on the current field and readings (see the table above for which can run on a narrower field from totals);
- `best(values)` with one candidate left;
- later: `runoff` (a real, separate runoff between two finalists, on two-round ballots), `btr(order, match)` (bottom-two climb), `repeat(...)` (eliminations, as INS).

**Chaining:** `>>` reads "then": the left side narrows the field (or transforms a reading), the right side works on what is left.

## Spelling (decided)

- `>>` for "then", `|` and `&` for sets.
- Plain names for readings without arguments (`score`, `margins`, `wv`), calls when there are arguments (`top(2, score)`, `soft(0.4)`).
- The kinds are told apart by a grammatical convention: readings are nouns for quantities (`score`, `margins`), sets are named for the set (`smith`, `top(...)`, `buck(...)`), transforms and final steps are verbs (`normalize`, `runoff`). The reader checks the kinds and names them in its messages (`top(2, smith)`: "top needs values, smith is a set").

## Examples

```
smith >> "Score"                                   Smith//Score
buck(points) >> normalize >> "Score"               ESR
buck(levels) >> "BTR-score"                        a new combination
ballots("rank") >> best(-worst(margins))           Minimax
best(share(score) - worst(wv) / n)                 MMWV+Score
top(2, score) >> runoff(score)                     Score runoff
(top(1, score) | challenger(top(1, score), margins)) >> runoff(score)     SCR (the add-on)
```

SCR shows why sets need union: the Score winner and their strongest challenger are two separate results of the same field, joined before the runoff. The repeated `top(1, score)` takes the same branch on both sides, so with a tie for the Score winner each tied winner meets their own challenger.

In an add-on the same text is Python, with a name:

```python
from compose import composed, smith, buck, points, normalize, score, wv, worst, share, best, n

METHODS = {
    "My ESR": composed(buck(points) >> normalize >> "Score"),
    "My mix": composed(best(share(score) * (1 - worst(wv) / n))),
}
```

and a function of its own can join in:

```python
def blend(s, w):                 # values in, values out
    return s - 0.5 * w

METHODS = {"Blend": composed(best(blend(share(score), worst(wv) / n)))}
```

On the command line: `python run.py --methods 'buck(levels) >> "BTR-score"' Score STAR`.

## Polls

Voters count a poll only from totals a real poll could show (model.md, What a poll shows), so every composed method has a poll view, built from its parts:

- if every reading of the composition is poll-like (per candidate, per pair, per grade), the composition counts its own polls;
- otherwise each part brings its poll reading: `normalize` is read as nothing in a poll (the raw scores), a quoted method as its own poll view (its stand-in, if it has one). Readings over every subset have no poll-like counterpart of their own (yet; see Decided in phase 1);
- a composition whose poll reading would itself need the exact fallback (a field-dependent reading after a set, also in the poll views) is polled by the lab's default poll reading: score totals on score ballots, first preferences otherwise, as an add-on method without a stand-in.

This gives the stand-ins written by hand today: ESR and INS are polled as Score because, without renormalising, the Score winner wins. Individual voters need the poll view to be countable from totals (closeness and the scenario table count the summed totals many times per round), so the exact fallback serves the election count only, never a poll.

## How it would run

- **From totals where possible.** The lab sums per-voter features, so a poll is a weighted sum, not a recount. A composition whose readings all restrict to a narrower field (the table above) compiles to an ordinary feature-based method.
- **Exact fallback** (built in phase 2). Otherwise each reading that needs it is counted again from the ballots restricted to the current field: scores and marks as cast (a vote for a candidate who is out counts for nobody), rankings as the induced ranking among the field (renumbered, ties kept, an unranked candidate stays unranked), two-round ballots both parts so; after `normalize`, the ballots stretched over the field it saw. The composition is then counted from the ballots (`Method.full`), exact but slower (`ballots("rank") >> smith >> "Borda"`: about 1.5 ms per count, Borda 0.02 ms); for the election count only (see Polls). The tests run compositions both ways where both exist and require the same result.
- **Ties and speed.** A tie splits the field into weighted branches; a count follows each branch and adds up the weights. Polls and the individual voters' closeness evaluate a count on many rows of totals at once, so a composed decision has to be cheap per row: no more than about twice the time of the hand-written method it reproduces. Measured: ESR 0.8×, Smith//Score 1.3×, Score runoff 1.6×, MMWV+Score 1.75×, MMPO 1.8×, Minimax 2.0×, Soft Minimax 2.1×, Approval runoff 2.2×; SCR 3.6× (the hand-written add-on is very compact; the overhead is the branching of `top(1, ...)` and `challenger` shared across `|`).
- **Ballot types.** Each part states the ballot types it works on; a composition has one, and a mismatch is reported with the part's name. `runoff` switches the composition to the two-round ballot.
- **Settings.** A composed method takes the settings of its final step (`Method.settings`, e.g. a fixed score range), if it has any.
- **Reuse of results.** A part that appears twice in one set expression takes the same branch of a tie (the repeated `top(1, score)` in SCR is the same Score winner on both sides). Sharing across methods would apply only to the honest count (with learned ballots every method has its own ballots), so it is not planned.
- **Existing methods** can then be written as compositions where that is exact. Each rewrite has to reproduce the hand-written version exactly (the regression check), so this can happen method by method.

## Names in the files

Results store the expression in a fixed canonical spelling (spaces and quotes normalised), so the same composition always has the same name and the reports can read it back. The canonical name is written to the CSV as an ordinary (quoted) cell; plots and other files named after a method use a safe file name derived from it (letters, digits and dashes, plus a short hash of the canonical name so that two compositions never share a file). The method fingerprints use the canonical name. A composition used in an add-on can be given a plain name instead (`"My ESR"`), which is then used everywhere.

## Groups in the plots

A composed method is placed in the group of its final quoted method (Smith//Score as composed: rated, with Score); one that ends in a set or `best(...)` goes into the group `composed`. An add-on can give a group (`GROUP`, as add-ons do now).

## Strategy (decided)

Voters see polls of the composed result as a whole, as for every method: no voter reasons about the parts. Both strategy engines need only the ballot type (the moves, the naive exaggerations) and the poll view, so nothing in them changes.

## Phases

1. **The core** (built 2026-11-01): the expression reader; sets, values and pairs with ties; `>>`, `|`, `&`; the state with pass-through and the field-dependence check; `normalize`; the parts `smith`, `top`, `best`, `buck`, `score`, `margins`, `wv`, `n`, `levels`, `points`, `share`, `worst`, `ballots(...)`; quoted methods whose features restrict; the automatic poll view; canonical names and safe file names; named compositions in add-ons. Checked in the tests: Smith//Score, ESR, MMWV+Score and Minimax written as compositions give exactly the hand-written methods, their polls included; in full runs (30 elections, all voters strategic) the same results in every row, except 2 of 1,440 values for ESR (see Polls). Speed per count against the hand-written method: ESR 0.8×, Smith//Score 1.3×, MMWV+Score 1.75×, Minimax 2.0×.
2. **More parts and runoffs** (built 2026-11-01): `po`, `soft(0.2 / 0.4)`, `approvals(k)`, `challenger(set, pairs)`, `runoff` and `runoff(values)`, pairwise readings on two-round ballots from the round-1 marks (a new feature `round1_gt`), and the exact fallback. Checked in the tests: Score runoff, Approval runoff, MMPO, Soft Minimax and SCR (the add-on) written as compositions give exactly the hand-written methods, polls included; Borda among the Smith set by the exact fallback equals Borda on the induced rankings; the totals and the exact fallback agree on five compositions.
3. Orders and iteration (`btr`, `repeat`), then rewriting existing methods where exact.
4. Multi-winner results.

## Decided in phase 1

- **ESR in phase 1:** `buck` came forward from phase 2, with two readings per level: `levels` (voters) and `points` (voters times the level's score). ESR is `buck(points) >> normalize >> "Score"`; Bucklin-like sets by voters are `buck(levels)`.
- **Readings over every subset in polls:** no generic poll-like counterpart. They are polled through the parts' own poll views: `normalize` is read as nothing, a quoted method by its stand-in. A counterpart per reading can be added when a part needs one.
- **The poll arithmetic depends on what a poll publishes.** A composition's poll view publishes every total its parts read. For ESR that includes the score levels (for `buck`), which the hand-written stand-in (Score) does not: the winners of every poll are the same, but individual voters project their moves on all published totals, so their answers can differ slightly (2 of 1,440 values in the trial). Left as it is: what a poll publishes is part of a method's description.
- **Sets at the end** split the win equally among their members; **an empty intersection** narrows nothing; **`|` binds looser than `&` and `>>`**, as in Python: `(best(score) | smith) >> "Score"` needs the parentheses.

## Decided in phase 2

- **`runoff` names its tie-break.** The hand-written two-round methods give a runoff tie to the higher round-1 value (Score runoff, Approval runoff); a composition says so: `runoff(score)`. Plain `runoff` splits a tie. With more than two finalists (a tie before the runoff), the candidates with most runoff wins win, ties split.
- **`challenger(set, pairs)`** is a set of its own (the strongest beaters of the set's members), joined with the set by `|`: `(top(1, score) | challenger(top(1, score), margins)) >> runoff(score)`. `top(1, ...)` rather than `best(...)` makes a tie for the Score winner branch, as SCR does: each tied winner with their own challenger. A set nobody beats has no challenger: an empty set, which narrows nothing in a union.
- **Ballot type with `runoff`:** score runoff ballots unless `ballots(...)` says otherwise; `score` then reads the round-1 scores (approvals or marks on the other two-round types), and the pairwise readings come from the round-1 marks.
- **The exact fallback** replaces the error for field-dependent readings after a set; the error remains only for ballot types that cannot be restricted to a field. Its polls use the lab's default poll reading when the poll view would need the fallback too.

## Open questions

- Phase 3 (orders and iteration: `btr`, `repeat`) and rewriting existing methods as compositions where exact.
