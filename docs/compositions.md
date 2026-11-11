# Compositions: methods built from parts

A voting method can be written as a composition of parts instead of as Python code: on the command line, or in an add-on. A composition runs like any other method, with every strategy engine, all measures and the reports.

```
python run.py --methods 'smith >> "Score"' 'best(share(score) - worst(wv) / n)' Score
```

Quote a composition on the command line (single quotes around it, double quotes around method names inside), as above. In an add-on the same text is Python:

```python
from compose import composed, smith, best, share, score, worst, wv, n

METHODS = {
    "My Smith//Score": composed(smith >> "Score"),
    "My mix": composed(best(share(score) * (1 - worst(wv) / n))),
}
```

A Python function that takes values and returns values can join in, since values can be added and multiplied like numbers:

```python
def blend(s, w):                 # values in, values out
    return s - 0.5 * w

METHODS = {"Blend": composed(best(blend(share(score), worst(wv) / n)))}
```

The code is in compose.py. Many built-in methods carry their composition as a comment in methods.py (`# composed: ...` above the method), for copying as a starting point (see Built-in methods as compositions, below). A few built-in methods are compositions themselves (`methods.BUILTIN_COMPOSED`, registered at start-up): ESR-MARS, `buck(points) >> "MARS"`.

## How a composition works

A composition is a chain of steps, joined by `>>` (read "then"). It works on two things: the **field**, the candidates still in, and the **readings** of the ballots known so far (score totals, the pairwise matrix, ...). Each step changes only what it is about:

- a **set** (`smith`, `top(2, score)`, ...) narrows the field and leaves the readings as they are;
- a **transform** (`normalize`, `max_top`, `zero_bottom`) replaces a reading; later steps see the new one;
- a **reading** or a **last step** takes the latest version of what it needs, on the current field.

So `smith >> "Score"` is: the score ballots; `smith` narrows the field to the Smith set and leaves the scores untouched; `"Score"` takes the same score totals, on the Smith set only, and the highest wins. That is Smith//Score.

**Readings that depend on the field.** Passing a reading on is exact when it does not depend on which other candidates are in: a candidate's score total, or how many voters prefer A to B. Narrowing the field just drops columns or rows. Other readings depend on the field: first preferences (whose favourite among those left?), rank points (Borda), scores stretched over the field. These are never taken from an earlier field. After a set they are counted again on the current field: from totals over every subset of candidates where the lab keeps them (first preferences among the field, scores stretched over it), else from the ballots restricted to the field (the exact fallback, below). So `smith >> "Plurality"` means Plurality among the Smith set, as it should.

## The parts

### Readings

| Part | Kind | Meaning |
|---|---|---|
| `score` | values | each candidate's score total. On rankings: rank points (Borda), which depend on the field. On two-round ballots: the round-1 scores, approvals or marks |
| `firsts` | values | first preferences among the current field (rankings). On rankings with ties, a whole vote for each candidate tied at the top (as ER-IRV); on truncated rankings, a ballot with nobody ranked among the field counts for nobody |
| `margins` | pairs | for A against B: voters preferring A minus voters preferring B (equal scores or ranks count for neither) |
| `wv` | pairs | winning votes: for A against B, the voters preferring A if A wins the pair, else 0 |
| `po` | pairs | pairwise opposition: for A against B, the voters preferring A, whoever wins the pair |
| `soft(f)` | pairs | soft margins (as Soft Minimax): per voter the score difference A − B, clipped to ±f of the score range and divided by f × range, summed over voters. 0 < f ≤ 1. With small f every difference counts as a full vote (majority margins); with f = 1 the score differences themselves, so the best worst soft margin is the Score winner |
| `n` | number | the number of voters |
| `held` | values | after `proxy`: the votes each candidate of the field holds. `repeat(proxy("score") >> drop(held))` is weakest first with candidates who pass on by their score ballots |
| `levels` | per level | voters per score level and candidate, top level first (score ballots) |
| `points` | per level | the same, each level weighted by its score (the top level of 0–5 counts 5 per voter, the next 4, ...) |
| `approvals(k)` | values | voters scoring a candidate at least k (score ballots) |
| `median` | values | each candidate's median score, the lower median as majority judgment reads it (score ballots). `best(median)` is MJ's set of median winners, before its tie-break |

On two-round ballots the pairwise readings come from the round-1 marks; the runoff rankings are read only by `runoff`.

### Arithmetic

| Part | Kind | Meaning |
|---|---|---|
| `share(values)` | values | values / (n × top score): a score total as a share of the most it could be |
| `worst(pairs)` | values | each candidate's worst defeat: the largest value any rival has against them, 0 if none is positive. `worst(margins)` is Minimax's measure |
| `total(pairs)` | values | each candidate's sum of values against all others. `total(po)` counts, over voters, how many candidates each voter puts below the candidate: on strict rankings, Borda points among the field |
| `diff(values)` | pairs | turns values into pairs: for A against B, value(A) − value(B). Used to mix a per-candidate measure into a pairwise match, as MARS does: `margins + n * diff(share(score))` adds to each margin the difference of the two score totals divided by the top score |
| `+ - * /`, `-x` | | values with values, pairs with pairs, each with numbers |

### Sets

A set narrows the field. A tie inside a set branches: each way the tie can go is followed with its weight, as the lab's methods do.

| Part | Meaning |
|---|---|
| `smith` | the Smith set: the smallest set of candidates each of whom beats or ties everyone outside it head to head |
| `schwartz` | the Schwartz set: the candidates who, if anyone has a path of pairwise defeats to them, have one back (the union of the smallest sets nobody outside defeats). Only strict defeats count, so ties can make it smaller than the Smith set; without ties the two are the same |
| `cdtt` | Woodall's CDTT set: the Schwartz set of majority defeats, where A defeats B only if more than half of all voters prefer A to B. Voters who rate the two alike or leave them unranked count against both, so with ties or truncated ballots the set is larger than the Schwartz set; with complete strict rankings and an odd number of voters the three sets agree |
| `banks` | the Banks set: the top candidates of the maximal chains, subsets in which every pair is decided and the defeats run in one order, that no further candidate can join (a tied pair cannot both be in a chain). Within the Smith set (within the uncovered set, for a complete tournament); counted over every subset of the field |
| `uncovered` | the uncovered set: the candidates nobody covers, where A covers B if A defeats B and everyone B defeats (strict pairwise defeats, Miller's definition). Within the Smith set; contains the Banks set (for a complete tournament) |
| `best(values)` | everyone tied for the highest value (no branching: the tied candidates stay together) |
| `top(k, values)` | the k highest; a tie at the k-th place branches with equal weights. `top(1, ...)` is "the highest", with a tie branching |
| `drop(values)` | everyone but the lowest; a tie for lowest branches (each of them dropped, equal weights) |
| `buck(levels)` | a Bucklin-style set by voters: for every depth k (the top 1, 2, ... score levels together), the candidates with the most voters scoring them in those top k levels; these leaders, joined over all depths, form the set. At depth 1 it has the candidates with the most top scores; at the full depth, those with the most voters giving them any score above 0 |
| `buck(points)` | the same with points: at every depth, the candidates with the most score points from the top k levels. At the full depth that is the score total, so the Score winner is always in this set. ESR is `buck(points) >> normalize >> "Score"` |
| `challenger(set, pairs)` | the strongest challengers of a set: the candidates outside it with the largest value (of the pairs) against one of its members. Ties branch. Nobody with a positive value: an empty set |
| `bottom_two(order, pairs)` | one round of bottom-two runoff: the two lowest by the order meet, and the loser of the match is out (pairs: a positive value means the row candidate wins). A tie in the match is split; `bottom_two(order, pairs, tiebreak)` gives it to the higher tiebreak value first |
| `btr(order, pairs)` | the bottom-two climb on a fixed order: the field ordered high to low, the lowest two meet, the winner meets the next one up, and so on; the last winner is the set. A tie in a match is split; with a tiebreak, `btr(order, pairs, tiebreak)`, it goes to the higher tiebreak value and, when those tie too, to the candidate higher in the order (as BTR-score). Tied orders: every order of the tied candidates, equally |
| `repeat(steps)` | the steps (sets and transforms) again and again, until one candidate is left or a round changes nothing. Each round starts from the readings before the repeat, so a `normalize` inside stretches the ballots over the field left, every round. Only what `proxy` does carries over from round to round: it changes who holds the votes, not how they are read |
| `"Name"` | an existing method, used as a set: the set of its winner on the current field (a tie in its result branches with the method's weights) |
| `a \| b`, `a & b` | the union and the intersection of two sets of the same field. An empty set narrows nothing. The same part twice in one set expression takes the same branch of a tie |

**`bottom_two` or `btr`.** With a fixed order and no ties, `repeat(bottom_two(order, pairs))` and `btr(order, pairs)` elect the same candidate. They differ in two ways: with `repeat`, the order is read again on the field left each round (BTR-IRV recounts first preferences after each elimination); and with ties in the order, a climb and repeated eliminations give the ties different weights. BTR-score and MARS climb; BTR-IRV eliminates round by round.

**A set's own zero point.** A set that narrows the field to good candidates lifts every method that follows it, and a random ballot among the set shows by how much: `smith >> random_ballot` or `buck(points) >> random_ballot` is the baseline of that set, as the random ballot is the zero of VSE and WME for the whole field. Methods after the set then spread again between this new zero and the same best candidate, so comparing `smith >> "X"` with `smith >> random_ballot` shows what X adds within the Smith set. (The pattern repeats only once for the Smith set: the Smith set of the Smith set is itself.)

**Methods as sets.** `("DMS" | "ESR") >> runoff(score)` is a runoff between the DMS winner and the ESR winner (the same candidate: no runoff needed). `("Score" | challenger("Score", margins)) >> runoff(score)` is SCR: the Score winner and their strongest challenger meet in a runoff.

### Transforms

| Part | Meaning |
|---|---|
| `normalize` | every score ballot stretched over the current field: its lowest candidate there 0, its highest the top score, the rest in proportion; not rounded. A ballot that rates the whole field alike counts 0 for all. Later steps read these scores (`score`, and a quoted method's score totals). Pairwise readings are unchanged, since stretching keeps each ballot's order |
| `max_top` | every score ballot's highest-rated candidates in the current field count the top score, the rest as given; a ballot rating the whole field alike is left as it is. The first step of IMDS |
| `zero_bottom` | every score ballot's lowest-rated candidates in the current field count 0, the rest as given; a ballot rating the whole field alike counts 0 for all. The second step of IMDS: `repeat(max_top >> zero_bottom >> drop(score))` is IMDS, `repeat(zero_bottom >> drop(score))` IDS (the add-on) |
| `proxy`, `proxy("type")` | asset voting ([methods.md](methods.md), Asset voting and delegation): every vote is given to candidates of the current field as the voter's ballot divides it (`methods.vote_shares`: one name or a ranking to the first choice; approval in equal parts to the approved, as a cumulative vote; scores and grades by the Kotze-Pereira transformation), and from then on the candidates cast it, each with their own sincere ballot of the type given. A bare `proxy` takes the ballot type of the composition's last quoted method, else rankings; the candidates' type must order more than one candidate (not one name, veto or two-round). The electorate is then the candidates, each weighted by the votes they hold, and every later reading counts their ballots. A candidate who is out of the field still holds their votes, and the readings follow that candidate's own ballot among the field. Inside a `repeat` the holdings carry over: a proxy in a later round passes everything a candidate out of the field holds on as their own ballot divides a vote, and candidates in the field keep what they hold. So `repeat(proxy >> drop(firsts))` passes piles by the holder's list, and `proxy >> repeat(drop(firsts))` by the list of the candidate a vote first went to. In a poll only the holdings come from the poll; what the candidates' ballots then give is counted exactly |

**Several transforms.** Transforms act in turn, each on the ballots as the one before left them and on the field at its step. `normalize` after any of them gives what `normalize` alone gives, since it stretches every ballot anew. Scores after `max_top` and `zero_bottom` keep each ballot's order, so the pairwise readings and first preferences are unchanged, as after `normalize`; readings by score level (`levels`, `points`, `approvals`, `median`) are not available after any transform. Score totals after the transforms come from totals over every subset where the lab keeps them (`normalize` alone; `max_top` and `zero_bottom` on the same field, as in IMDS), else from the ballots (the exact fallback).

### Last steps

| Part | Meaning |
|---|---|
| `"Name"` | an existing method, on the current field and readings: its winner lottery |
| `runoff` | a separate runoff between the finalists, decided by the runoff rankings of two-round ballots; a tie is split. With more than two finalists (a tie before the runoff), those who win most runoffs win, ties split |
| `runoff(values)` | the same, but a tie goes to the higher value (as Score runoff and Approval runoff: `runoff(score)`) |
| `random_ballot` | a ballot drawn at random, then a candidate of the field in proportion to what the ballot gives them: on score, grade and approval ballots and single marks the values themselves (on two-round ballots the round-1 marks), on rankings the first preference among the field; a ballot giving the field nothing is drawn again. Not on veto ballots. In a poll, in proportion to the totals. The built-in method `"Random ballot"` is the same on score ballots |
| a set | a composition may end in a set: the win is split equally among its members, so `best(values)` at the end means "the highest wins, ties split" |

### Ballot type

`ballots("type")` as the first step sets the ballot type (the types are listed in ballots.py: `score`, `rank`, `weak`, `strict`, `trunc`, `fav`, `approval`, `score_run`, ...). Without it: with a `runoff`, the two-round ballot whose first round fits the first quoted method (score runoff ballots for a method on score ballots, approval runoff for approval, two-round plurality for one name; score runoff ballots without a quoted method); otherwise the ballot type of the first quoted method; otherwise score ballots.

On two-round ballots, a quoted method of the first round's type counts the round-1 marks: `"DMS"` on score runoff ballots is DMS on the round-1 scores.

`delegate` as the first step (or right after `ballots(...)`) makes the ballots delegated: what a lazy voter leaves blank is filled from their favourite's own ballot, below everything the voter marked ([methods.md](methods.md), Asset voting and delegation). On rankings (`rank`, `weak`), scores and three grades; without laziness (`--lazy 0`) it changes nothing. `delegate >> "321"` is Delegated 321, `delegate >> "Score"` delegated Score.

## Polls

Voters see a poll only as totals a real poll could publish ([model.md](model.md), What a poll shows), so every composition has a poll view, built from its parts:

- if every reading of the composition is such a total (per candidate, per pair, per score level), the composition counts its own polls;
- otherwise each part brings its poll reading: `normalize`, `max_top` and `zero_bottom` count as nothing (the raw scores); a quoted method counts by its own poll view (its stand-in, if it has one); `firsts` is estimated from the first preferences and the transfer table, as the IRV family's stand-in;
- a composition whose poll would itself need the exact fallback is polled by the lab's default poll reading: score totals on score ballots, first preferences otherwise.

What a poll publishes also shapes the voters' arithmetic: individual voters project their moves on every published total. ESR written as a composition publishes the score levels (which `buck` reads) besides the score totals, while the hand-written ESR's stand-in publishes the score totals only. The poll winners are the same, but voters can answer slightly differently (2 of 1,440 values in a trial).

## The exact fallback

A reading that depends on the field, needed after a set where the lab keeps no totals over every subset for it, is counted again from the ballots restricted to the field:

- scores and marks as cast; a vote for a candidate who is out counts for nobody;
- rankings as the induced ranking among the field: renumbered, ties kept, an unranked candidate stays unranked;
- two-round ballots both parts so;
- after `normalize`, the ballots stretched over the field it saw.

Such a composition is counted from the ballots, which is exact but slower: `ballots("rank") >> smith >> "Borda"` takes about 1.5 ms per count, Borda 0.02 ms. It is used for the election count; polls use the poll view (above). For a ballot type that cannot be restricted (an add-on's own type), the composition is refused with a message naming the reading.

## Names, groups, settings

- **Names.** A composition is stored under a canonical spelling, so `smith>>"score"` and `smith >> "Score"` are the same method. Files about one method (track plots) get a safe name with a short hash; reports escape `|`, `*` and `_` in names. In an add-on, a composition gets the name it is given.
- **Group** (the plot panels): the group of the last quoted method; a composition that ends otherwise goes into the group `composed`.
- **Settings.** A composition takes the settings of its last quoted method (e.g. a fixed score range).
- **Precedence.** `|` binds more loosely than `>>`, as in Python, so a union before `>>` needs parentheses: `(top(1, score) | smith) >> "Score"`.

## Built-in methods as compositions

These compositions give exactly the built-in methods, polls included (checked by the tests, which read the comments in methods.py):

```
smith >> "Score"                                         Smith//Score
buck(points) >> normalize >> "Score"                     ESR
best(share(score) - worst(wv) / n)                       MMWV+Score
ballots("rank") >> best(-worst(margins))                 Minimax
ballots("weak") >> best(-worst(po))                      MMPO
best(-worst(soft(0.2)))                                  Soft Minimax
best(-worst(soft(0.4)))                                  Soft Minimax-40
top(2, score) >> runoff(score)                           Score runoff
ballots("app_run") >> top(2, score) >> runoff(score)     Approval runoff
ballots("rank") >> repeat(drop(firsts))                  IRV
ballots("trunc3") >> repeat(drop(firsts))                IRV-3
ballots("weak") >> repeat(drop(firsts))                  ER-IRV
ballots("rank") >> repeat(drop(total(po)))               Baldwin
repeat(normalize >> drop(score))                         INS
repeat(max_top >> zero_bottom >> drop(score))            IMDS
ballots("rank") >> repeat(bottom_two(firsts, margins))   BTR-IRV
btr(score, margins, score)                               BTR-score
btr(score, margins + n * diff(share(score)))             MARS
ballots("fav") >> repeat(proxy >> drop(firsts))          Asset
delegate >> "321"                                        Delegated 321
repeat(zero_bottom >> drop(score))                       IDS (the add-on)
("Score" | challenger("Score", margins)) >> runoff(score)                  SCR (the add-on)
```

The built-in methods stay hand-written, because a composition takes longer per count: about 1.4 to 3 times as long, measured on these.

## Design notes

**Why.** Many methods are built from the same parts: a narrower field (Smith//Score: the Smith set, then the highest score), a measure per candidate (score totals, worst defeats), a way to combine measures (MMWV+Score adds a score share and a defeat), a last step (the highest value, a bottom-two climb, a runoff). Written by hand, the parts cannot be reused, and two methods sharing a part may differ in small ways nobody intended. Named parts make the differences visible and new combinations cheap.

**Principles.**

1. One notation everywhere: the command line and add-ons use the same text, a small subset of Python expressions. The command line reads it with Python's own parser and accepts only the known parts; no other code runs.
2. Names stay names: existing methods appear as quoted strings. A method called `A//B` is never read as a composition, and the literature's `A//B` (which often differs from the lab's A and B in detail) is not an operator.
3. Parts are values, not only steps: values and pairs can be added, multiplied and passed to Python functions.
4. Exact, with ties: every result is a weighted list of alternatives.
5. No change to the rest of the lab: a composition becomes an ordinary method (ballot type, features, decision, poll view, settings).

**How the features narrow.** Every built-in method decides from sums of per-voter ballot features (methods.py, `FEATURES`). A composition reads them on a narrower field like this:

| Kind | Features | On a narrower field |
|---|---|---|
| per candidate, independent of the field | `sum` (score, grade and approval ballots), `app`, `run_sum`, `maxs`, `pos`, `upper`, `ge1`, `ge2`, `eq0`, `levels`, `grades` | the columns of the field |
| per pair, independent of the field | `gt`, `runoff_gt`, `round1_gt`, `excess`, `soft20`, `soft40` (and other soft margins), `levelled`, `diffhist` | the rows and columns of the field |
| per subset, among it | `firsts`, `firsts_eq`, `firsts_ex`, `lasts`, `keepnorm`, `nscore_sub` | the entries of the field's subsets (exact; not what a poll shows) |
| dependent on the field | `first`, `last`, `top`, `eqtop`, `eqbot`, `first_split`, `transfer`, `transfer_whole`, `lasttransfer`, `favscore`, `dowdall`, `firstsecond`; `fset`, `fsetB`, `solid`, `acquiesce` (per subset, but judged against everyone outside it); `sum` on anything but rated ballots (rank points, marks) | the exact fallback |
| always | `n` | unchanged |

Features read on the round-1 marks of two-round ballots (`r1_...`, made when a quoted method of the first round's type needs them) narrow as the feature they are made from.

**Decisions along the way** (details in [journal.md](journal.md), 2026-11-01):

- Values after a set are passed through and restricted, never silently recomputed; a reading that depends on the field is counted again on it, never taken from an earlier field.
- `runoff` and `btr` name their tie-break (`runoff(score)`, `btr(score, margins, score)`) rather than hiding it.
- `top(1, ...)` branches on a tie, `best(...)` keeps the tied together: SCR needs the first (each tied Score winner with their own challenger).
- Readings over every subset have a poll-like counterpart only where one exists in the lab (`firsts`, by the IRV stand-in); otherwise the parts' own poll views decide.
- The built-in methods are not rewritten as compositions, since compositions are slower; the comments in methods.py are the reference for copying.

**Next.** Multi-winner results (committees) with the same machinery are planned for a later version: a composition would end in a set of k winners instead of one.
