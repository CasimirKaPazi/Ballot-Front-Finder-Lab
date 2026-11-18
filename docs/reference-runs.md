# Reference Runs

A fresh set of runs with version 3.0.0 (individual voters as the default strategy engine, finding their moves by pooled discovery and each weighing the value of a move against a personal cost of voting insincerely; candidates chosen among the voters; the field as drawn by default, strategic nomination as a family of conditions), for comparison and as the base for new tests: the defaults, a large base, the main robustness tests, strategic nomination in several forms and a vse-sim-like setting. This file says how to make them again. It can also serve as the README of a repository of the results.

The set is chosen to show how robust the findings are, not to tune the model: the tests that only served to set the model's defaults (noise strength, dimensions and the like; see [journal.md](journal.md)) are not repeated. The parameters of the strategy engine are tested, since individual voters depend on them directly: the poll error sets how flat the chances are, the coordination how many of their camp a voter expects to act alike, and the honesty cost how much a move must gain, and the number of polls how noisy the voters' picture of the chances is.

Strategic nomination is not part of the base. Its model is crude (nothing fixes the cost of a move, how often candidates reconsider, how far they foresee the result), and it changes the results a lot: most methods move towards the Condorcet corner, the methods that attract clones away from it. So it is run as a family of conditions from none to cheap moves in every round, and a method is read by the area its results span across them (`compare.py --trace`), with the conditions at the corners as the extremes.

## Before Starting

- **One commit of the lab for everything.** Run all commands from the lab's folder at one commit and record it with the results:

  ```
  R=../bffl-runs                            # where the results go (a folder of its own)
  mkdir -p $R && git rev-parse HEAD > $R/lab-commit.txt
  python tests/test_methods.py && python tools/regression.py
  ```

  Every results row also records the simulator version and a fingerprint of each method's results ([running.md](running.md), Safety), so `compare.py` can tell later whether runs belong to the same code.
- **Add-ons.** Move the test add-ons out of `addons/` (keep `ids.py` and `_template.py`), so the runs contain only the lab's methods.
- **Methods.** The tests use the set `extended`: the standard set, the strongest newer methods (ESR, DMS, Soft Minimax, Soft Minimax-40, MARS-T, IMDS, ESR-MARS) and the delegated methods Asset and Delegated 321 (38 methods), so every test shows them in all conditions. Delegated 321 differs from 321 only with laziness (the laziness and hard tests). The default run uses the plain standard set, as a user gets it.
- **The same elections everywhere.** All runs use seed 1, so a test that changes only strategy or ballots faces exactly the electorates of the base. The base has 4000 elections and every test 2000 (at least): a test's elections are the base's first 2000, and the base's other 2000 make the reference more precise. Since the tests are smaller than the base, the rank stability table has a second noise row for them ([analysis.md](analysis.md), rank stability).
- **Time it first.** Run the base with `--elections 20`, and scale: time grows with the elections, the methods, the candidates and the poll rounds. As a guide, the cost of each item is given below in **units**: one unit is 1000 elections of the extended set at the defaults (about the time of `--elections 20` × 50, less the start-up of about 6 s for the fingerprints; about 9 minutes on two cores). Measured on 20 elections each (2026-11-07, version 2.1.0, then with strategic nomination; without it a run is somewhat faster, and nomination in the race adds about half again at 3 poll rounds: estimates, to be timed). The costliest settings: 10 candidates (about 5 units per 1000 elections; 8 candidates about 2), the ranking groups (2.5, with all voters strategic 4.5) and the arithmetic response needed for campaigns (1.5 to 2).

## The Runs

```
# 1. Reference                                                            units
python run.py --out $R/defaults                                          # 0.1   200 elections, standard set
python run.py --elections 4000 --methods extended --manipulation --out $R/base           # 5
python run.py --elections 4000 --methods all --exclude extended inverse \
    --out $R/base-rest                                                   # 5     every other method and the references, same elections
python sweep.py --model ks-quinn --utility linear --strategic 0.5 1 --adaptive 0 --rounds 1 \
    --voters 40 --elections 15000 --max-score 10 --exits 0 --clones 0 --cands-vote 0 --methods 321 STAR Score \
    Approval "Ranked Pairs" Schulze IRV MJ MAV Borda Plurality --out $R/vsesim   # 1.8

# 2. Strategy
python sweep.py --strategic 0.5 1 --adaptive 0 --elections 2000 --methods extended --out $R/strategic-share  # 3.7
python sweep.py --coordination 0 0.1 0.5 --elections 2000 --methods extended --out $R/coordination        # 5.7
python sweep.py --honesty 0.1 1 --elections 2000 --methods extended --out $R/honesty                     # 4
python sweep.py --poll-error 1 4 --elections 2000 --methods extended --out $R/poll-error                   # 4
python sweep.py --samples 10 20 --elections 2000 --methods extended --out $R/samples                      # 6
python sweep.py --rounds 1 2 4 5 --elections 2000 --methods extended --out $R/rounds                       # 2.7 (one run)
python run.py --start previous --elections 2000 --methods extended --out $R/lockin                         # 2.4
python sweep.py --campaign 0 1 --engine arithmetic --strategic 0 1 --elections 2000 --methods extended --out $R/campaigns   # 14
python sweep.py --engine groups --strategic 0 1 --elections 2000 --methods extended --out $R/groups         # 14

# 3. Ballots and candidates
python sweep.py --lazy 0.25 0.5 --elections 2000 --methods extended --out $R/laziness                        # 4
python run.py --max-score 99 --elections 2000 --methods max-score --exclude inverse references --out $R/max-score   # 2.5
python run.py --exits 1 --clones 2 --race-cost inf --elections 2000 --methods extended --out $R/nomination-before   # 2
python sweep.py --exits 1 --clones 2 --race-cost 0.05 0 --elections 2000 --methods extended --out $R/nomination-race  # 6
python run.py --exits 1 --clones 2 --race-cost 0 --rounds 6 --elections 2000 --methods extended --out $R/nomination-free  # 6
python run.py --exits 1 --clones 0 --elections 2000 --methods extended --out $R/nomination-parts/exits      # 3  (optional)
python run.py --exits 0 --clones 2 --elections 2000 --methods extended --out $R/nomination-parts/clones     # 3  (optional)

# 4. Electorates
python sweep.py --model clusters field ks --elections 2000 --methods extended --out $R/models                # 6
python run.py --width 1.4 --elections 2000 --methods extended --out $R/width                                 # 2
python sweep.py --utility linear quadratic --elections 2000 --methods extended --out $R/utility              # 4
python sweep.py --cand-dispersion 0.5 2 --elections 2000 --methods extended --out $R/cand-dispersion         # 4
python sweep.py --cand-shift 0.5 1 --elections 2000 --methods extended --out $R/cand-shift                   # 4
python sweep.py --cands 4 8 10 --elections 2000 --methods extended --out $R/candidates                       # 15.5 (10 candidates: 10.5)
python run.py --voters 199 --elections 2000 --methods extended --out $R/voters                               # 2.2

# 5. Combined
python run.py --lazy 0.25 --cand-dispersion 0.5 --poll-error 4 --elections 2000 --methods extended \
    --out $R/hard                                                        # 2.2
```

About 130 units in all (about 19 hours on two cores at the time of measuring; the units of the nomination runs are estimates until timed), a third of them for campaigns, groups and candidates; the analysis below adds about one to two hours. If that is too long, these come first (about 50 units): the defaults, the base, strategic share, coordination, honesty, poll error, groups, nomination before and in the race, the utility fall-off (width and utility), models and the vse-sim setting; the rest can follow. Every run can be interrupted and continued with the same command plus `--resume`; the same way, a finished run can be extended with a larger `--elections`.

**Left out, from the results of version 1.0.0** (findings.md of those runs): smaller polls (only the methods that strategy repairs, which the poll error test covers), 49 voters (the base's order of methods, as from noise), scores 0–9 (no clear change; the effect is at 0–99), width 2.8 (the same as width 1.4 and linear utility, further along), laziness 0.1 (hardly a change), 5, 7 and 9 candidates (the trends are monotone, the endpoints carry them), poll rounds beyond 5 (the result settles after the first rounds). Also left out: random variations of the moves (`--search`, no effect beyond noise in their first trial, journal 2026-10-31) and a lock-in test with nomination (the lock-in test runs on the field as drawn, like the base; nomination has its own family of runs).

What each part answers:

| Part | Question |
|---|---|
| defaults | what a user gets with `python run.py`: the example to compare one's own runs with |
| base, base-rest | every method on the same 4000 elections at the defaults (the reference methods too), with the manipulation test; the reference for all tests |
| vsesim | close to vse-sim: Quinn's hierarchical model as in its code, linear utility, 40 voters, scores 0–10, a fixed share of strategic voters answering one poll, a fixed field of candidates apart from the voters; vse-sim's methods |
| strategic-share | a fixed share of always-strategic voters (half, all) instead of adaptive voters |
| coordination | individual voters expecting nobody but themselves (0), a tenth or half of their camp to vote as they do, instead of a quarter (the base) |
| honesty | the cost of voting insincerely up to a tenth or the whole of a voter's utility range instead of 0.3: about twice and a third as much strategy |
| poll-error | voters treating polls as exactly as precise as their sampling error (1) or four times less precise (4), instead of twice: the poll error sets how flat the chances are, and so how much a move gains |
| samples | 10 and 20 polls per round instead of 5, each with the same spread. With pooled discovery the voters' picture of the chances becomes finer: with few polls each move's value is a coarse estimate, and voters taking the best of several such values overreact to lucky ones (journal, 2026-11-06). The base keeps 5 on purpose, since overreacting to noisy information is part of what methods must withstand; this test shows which methods are sensitive to it. |
| rounds | one, two, four and five poll rounds (three in the base), all from one run: does the result settle, now that voters judge their moves from round to round? |
| lockin | the first poll from a previous election's strategic result (its ballots carried over to a field that can differ): can the electorate leave a strategic state? |
| campaigns | candidates recommending moves to their audience, with adaptive and with all voters strategic: how much coordination by campaigns changes. Campaigns need the arithmetic (`--response arithmetic`), so the test brings its own baseline without campaigns (campaign 0), with adaptive and with all voters strategic |
| groups | the ranking groups (the engine of version 1.0.0) instead of individual voters, at the defaults and with all voters strategic: how much the findings depend on the strategy engine |
| laziness | truncated rankings, fewer approvals and scores: a moderate (0.25) and a strong (0.5) cut |
| max-score | the score range 0–99 against 0–5: how much the coarse scale costs |
| nomination-before | strategic nomination before the race only (as the defaults from 2026-11-03): candidates withdraw, re-enter or field clones when the result with sincere ballots favours them |
| nomination-race | nomination also between the poll rounds, judged by the polls: at a cost of 0.05 of a candidate's utility range per move, and at no cost |
| nomination-free | no cost and 6 poll rounds: enough rounds for every candidate to leave if it pays, the far end of the family |
| nomination-parts | (optional) exits only and clones only, in the race: which of the two moves a method's results |
| models | three other electorate models: clusters (close to the base's), field and ks (Quinn's model as in vse-sim's code is in the vsesim setting) |
| width, utility | how quickly utility falls with distance: the default (mode-like), wider (towards linear), linear (median) and quadratic (mean) |
| cand-dispersion | candidates clustered in the centre or spread out, unlike the voters (since 2026-11-03 by who stands among the voters: mostly the most central or most extreme voters, see [model.md](model.md), Candidates who vote) |
| cand-shift | candidates half or one standard deviation of the voters further along the main dimension: candidates from one side (by who stands) |
| candidates | 4, 8 and 10 candidates (6 in the base), 2000 elections each; the base's 6 candidates are the first 6 of 8 and 10, with the same voters (nested electorates), so the comparison is paired |
| voters | 199 voters instead of 99 (polls of half the electorate): does the electorate's size matter, now that voters expect a share of their camp to act alike? The base's 99 voters are the first 99 of these 199 (nested electorates), so the comparison is paired |
| hard | several hard assumptions at once: candidates clustered in the centre, laziness 0.25 and poor poll information (poll error 4) |

## Analysis

```
for t in strategic-share coordination honesty poll-error samples rounds lockin campaigns groups laziness nomination-before nomination-race nomination-free nomination-parts/exits nomination-parts/clones models width utility cand-dispersion cand-shift candidates voters hard; do
    python analyze.py $R/base $R/$t                   # -> $R/$t/analysis-vs-base/
done
python analyze.py $R/base                             # -> $R/base/analysis/: the base on its own (manipulation test)
python analyze.py $R/defaults                         # -> $R/defaults/analysis/
python analyze.py $R/base $R/base-rest --out $R/overview-all-methods
python analyze.py $R/base $R/base-rest $R/max-score --show max-score --hide inverse references --out $R/max-score
python analyze.py $R/vsesim                           # -> $R/vsesim/analysis/
python analyze.py $R/base $R/width $R/utility --out $R/overview-utility
python show_electorates.py --sav --out $R/electorates          # -> $R/electorates/: the models' electorates without methods (add --model ... for the models to show)
python compare.py $R --pairs "Approval:Plurality" "Supplementary vote:Plurality" "TTR:Supplementary vote" \
    "IRV:TTR" "IRV-3:TTR" "Approval cond. runoff:TTR" "Approval cond. runoff:Approval" \
    "Approval runoff:Approval cond. runoff" "BTR-score:IRV" "IMDS:INS" "BTR-score:Smith//Score"
    --trace Plurality TTR IRV "Score runoff" Score STAR Schulze "Soft Minimax-D" BTR-score Approval --trace-together
                                                      # -> $R/comparison/: every method over all conditions, with the pairs
                                                      #    and the traced methods' areas across the conditions
```

`compare.py` counts every condition once and refuses runs of different code (fingerprints), so it should run cleanly on this folder. It uses the same bootstrap draws as analyze.py, so its dominance calls agree with the single reports. To read the vse-sim numbers against vse-sim's own, note the different zero point: the lab's VSE is 0 for a random ballot, vse-sim's for a random candidate ([analysis.md](analysis.md), Axes; the report gives a random candidate's value on the lab's scale).

## What to Publish

The results files are large: about 40 MB per 1000 elections of the extended set, more for sweeps over poll rounds (one set of rows per number of rounds) and in all about 6.5 GB. A results repository can hold this file, `lab-commit.txt`, the `defaults` run complete (its `elections.csv` is about 7 MB), and of every other run its `command.txt` and its analysis folders (`report.md`, `summary.csv`, plots), plus `comparison/` and the `overview-*` folders. With the commit and the commands here, anyone can make the full results again: every election depends only on its parameters, seed and number, so the same commands give the same numbers.

A `.gitignore` for that:

```
*/elections.csv
*/runs/
!defaults/elections.csv
```
