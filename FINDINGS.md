# Voting Pareto Lab — findings

What the simulations have shown so far. The model, parameters and definitions are in [README.md](README.md). Newest first; each section names the settings it used, since the model and defaults changed along the way (plain dimensions before Zipf, CE as an axis before WM). Numbers are point estimates; differences only count where the paired comparison or the noise reference says so.

**Units.** Up to 2026-10-06 the axes were PE (zero = utility-proportional lottery) and WM (the winner's worst margin in % of voters); the sections below keep those units. Since then they are VSE and WME, both with 0 = random ballot and 100 = best candidate (README, Axes). Within one condition the change is a fixed rescaling of each axis (the sums over elections that set the zero and the 100 are the same for all methods), so method order, dominance and rank stability are unaffected; only the numbers differ.

**Default poll rounds and version.** From 2026-10-10 the default is 3 poll rounds (before: 1), and every row records the simulator version (1.0.0 from then on). Sections before this note report round 1 unless they say otherwise.

**Output folders.** From 2026-10-11 every run or sweep writes into its own folder (`output/<out>/elections.csv`, analysis in `output/<out>/analysis/`). The commands in the sections below are from before: there `--out x` wrote `output/x-elections.csv`, and `analyze.py` was given that file. The same commands now write the folder, and `analyze.py x` reads it.

**Polls per round.** From 2026-10-11 the default is 10 polls per round (`--samples`, before: 30). A test with 1 to 30 samples showed systematic shifts only at 3 or fewer; from 5 up the results moved by a few points without a trend, which is strategy noise that more elections average out. 10 samples take about 37% of the time of 30. Sections before this note used 30.

**Removed methods.** On 2026-10-09 the Dowdall variants (Black-, BTR-, Baldwin-, Nanson-Dowdall; plain Dowdall stays) and the contingent vote (Contingent vote, Contingent-ER) were removed: the Dowdall test had answered its question (see the current picture below), and the contingent vote only served to separate the automated from the real runoff. Sections below still report them.

**Front.** From 2026-10-09 the analysis judges dominance on VSE*, WME* and sincerity: the result with strategic voters and how honest they can afford to be. Honest VSE and WME depend more on modelling assumptions and are context only. Earlier sections used other axes.

**Names.** On 2026-10-07 some methods were renamed: Top-two runoff → TTR, Approval runoff if needed → Approval cond. runoff (called ACR below), Minimax-wv → MMWV. The sections below use the new names.

## First look: Soft Minimax (2026-10-10)

Run (Claude, small): 100 elections, defaults, poll rounds 1 and 3; Soft Minimax with the clip at 20%, 40% and 60% of the score range, against Score, STAR, IC-MARS, INS, Minimax and MMPO. Point estimates; most differences are ties at this size. VSE* / WME* / sincerity at round 3:

| Method | VSE* | WME* | Sincerity |
|---|---|---|---|
| Score | 92.3 | 89.5 | 99.9 |
| Soft Minimax-60 (not kept) | 91.6 | 90.3 | 99.8 |
| Soft Minimax-40 | 89.7 | 93.0 | 99.8 |
| STAR | 89.4 | 94.3 | 99.7 |
| Soft Minimax (20%) | 89.1 | 97.2 | 99.8 |
| INS | 87.8 | 95.4 | 99.5 |
| Minimax | 82.4 | 100.0 | 97.0 |
| MMPO | 82.3 | 98.5 | 98.1 |

- **The clip moves the method along the trade-off,** from Minimax-like (20%) to Score-like (60%), as designed.
- **Soft Minimax (20%) reached the front at round 3** (with STAR and Minimax): VSE* tied with STAR, WME* about 3 points higher, as sincere as the score methods.
- **On the default 0–5 scale it is Minimax on score ballots:** a 1-point clip turns every integer score difference into a whole vote. So most of its advantage over ranked Minimax (VSE* 89 against 82, sincerity 99.8 against 97.0) comes from the ballot, not the rule: sincere score ballots leave close candidates level, and strategic voters reverse far fewer preferences on scores (burial 2.9% against 12.1%, compromise 0.3% against 7.6%). With a wider score range the 20% clip makes it a genuinely soft rule.
- Kept: Soft Minimax (20%) and Soft Minimax-40. Needs a full-size run.

## New methods: DSC, DAC, STLR, INS, MAMPO, COWPEA Lottery (2026-10-10)

Run (user): `sweep.py --one-at-a-time --strategic 0.5 1 --aggressive 0.5 0 1`, 400 elections, one poll round, defaults otherwise; the new methods with STAR, Score, Approval, MDDA, MMPO, Plurality and Random pair runoff for reference. INS was then still called "Iterated normalized score". Base condition, VSE* / WME* / sincerity, and paired differences:

| Method | VSE* | WME* | Sincerity |
|---|---|---|---|
| INS | 90.8 | 95.0 | 99.7 |
| Score | 90.4 | 93.4 | 99.9 |
| STAR | 90.3 | 94.9 | 99.8 |
| MDDA | 90.2 | 91.4 | 99.6 |
| MAMPO | 90.1 | 91.8 | 98.9 |
| Approval | 89.6 | 89.6 | 99.9 |
| DAC | 88.2 | 88.3 | 97.6 |
| DSC | 87.4 | 86.9 | 97.1 |
| STLR | 86.9 | 95.6 | 99.8 |
| MMPO | 86.7 | 99.7 | 97.7 |
| Random pair runoff | 54.1 | 43.9 | 98.3 |
| COWPEA Lottery | 27.2 | 26.3 | 99.6 |

- **INS performs well:** tied with Score and STAR on VSE* and WME* in every condition, slightly less sincere than Score. Honestly it trades 3.6 points of VSE for 4.1 of WME against Score and has clearly higher Condorcet efficiency (CE +8, CE* +4 to +8): renormalizing after each elimination recovers pairwise information that plain Score ballots flatten.
- **STLR loses VSE against STAR** (VSE −4.2 honest, VSE* −3 to −6 in every condition) and ties on WME and WME*: levelling the runoff to the preferred finalist favours the more widely preferred over the more strongly liked candidate.
- **MAMPO is tied with MDDA and Approval** on VSE* and WME*, a little less sincere; honestly it loses WME (−3.4) and CE (−5.4) against MDDA.
- **DAC is at least as good as DSC,** better under full strategy and with all-aggressive voters (WME* +2 to +4, CE* +3 to +6): with strict sincere ballots they are identical, and the difference comes from how strategic voters use equal rankings. Both stay in the middle of the field, with burial shares of 16–39%.
- **COWPEA Lottery is a weak lottery reference** (VSE* 27, below Random pair runoff's 54), and strategy lowers it further (honest VSE 36).
- The front (dominated by none on VSE*, WME*, sincerity) is Score and MMPO in three conditions, plus STAR with only kind voters.

## Candidates unlike the electorate (2026-10-10)

Run (user): `sweep.py --one-at-a-time --cand-dispersion 1 0.25 0.5 0.75 1.5 --cand-shift 0 0.5 1`, 200 elections, one poll round, defaults otherwise (strategic share 0.5, aggressive share 0.5), most methods. Voters are the same in every condition; only the candidates are moved (README, Electorates). Condorcet efficiency with honest ballots, which keeps its meaning across conditions (Condorcet methods: 100 by construction):

| Method | dispersion 0.25 | 0.5 | 0.75 | 1 (base) | 1.5 | shift 0.5 | shift 1 |
|---|---|---|---|---|---|---|---|
| STAR | 66 | 74 | 75 | 86 | 96 | 90 | 90 |
| MARS | 67 | 72 | 75 | 86 | 94 | 90 | 89 |
| Score | 55 | 64 | 65 | 77 | 88 | 80 | 86 |
| Approval | 52 | 57 | 67 | 78 | 85 | 75 | 67 |
| MJ | 47 | 51 | 60 | 69 | 83 | 73 | 77 |
| Borda | 87 | 83 | 86 | 89 | 89 | 81 | 76 |
| IRV | 33 | 36 | 49 | 65 | 87 | 73 | 83 |
| TTR | 33 | 31 | 46 | 68 | 87 | 75 | 89 |
| Plurality | 22 | 21 | 30 | 45 | 67 | 54 | 69 |
| Coombs | 90 | 94 | 97 | 99 | 99 | 98 | 99 |
| Condorcet winner exists | 82% | 93% | 97% | 96% | 99% | 97% | 99% |
| … with the highest total utility | 51% | 52% | 58% | 66% | 83% | 75% | 79% |

With strategy (VSE* / WME*), selected:

| Method | dispersion 0.25 | 0.5 | 1 (base) | 1.5 | shift 1 |
|---|---|---|---|---|---|
| STAR | 86 / 80 | 85 / 85 | 92 / 94 | 95 / 98 | 93 / 96 |
| Score | 87 / 74 | 89 / 84 | 91 / 92 | 94 / 97 | 95 / 95 |
| Approval | 77 / 72 | 80 / 78 | 90 / 89 | 93 / 94 | 87 / 85 |
| IRV | 75 / 47 | 79 / 62 | 87 / 84 | 93 / 94 | 91 / 96 |
| TTR | 72 / 49 | 78 / 63 | 87 / 86 | 93 / 97 | 91 / 98 |
| Minimax | 83 / 91 | 80 / 97 | 87 / 99 | 94 / 100 | 91 / 99 |
| Borda | 76 / 82 | 78 / 88 | 81 / 90 | 87 / 91 | 82 / 84 |

- **Clustered candidates hurt every method that is not a Condorcet method,** not only plurality and IRV as in Merrill (1984): from dispersion 1 to 0.25, Condorcet efficiency falls by about 20 points for STAR, MARS, Score, MJ and Approval and by 23–35 for IRV, TTR and Plurality. The order of the methods stays: hybrids, then rated and approval methods, then the IRV family, then plurality. Borda is nearly unaffected (89 → 87), Coombs almost so (99 → 90).
- **The electorates themselves get harder:** with clustered candidates a Condorcet winner exists less often (82% at 0.25), it is less often also the candidate with the highest total utility (51% against 66%), and choosing between the two costs more VSE (16–19% against 13%). Candidates who differ little give voters flat preferences (favourite ÷ least liked 1.5 against 4.3), so small differences decide, which plurality-type first choices and range-stretched scores both handle badly.
- **Under strategy the plurality-based methods suffer most:** IRV and TTR fall to WME* 47–49 at dispersion 0.25 (base 84–86); the rated methods hold 72–80 and the Condorcet methods about 90. Minimax's WME* dips most at 0.25 (91) and recovers at 0.5 (97).
- **Spread-out candidates (1.5) make every method better,** and the differences between methods shrink (CE 84–99 except plurality).
- **A shift of 0.5–1 voter SD makes elections slightly easier** (a Condorcet winner more often, and more often also the utilitarian best) and changes the order little; plurality-type methods gain. Approval loses at shift 1 (CE 78 → 67, VSE* 90 → 87), where many voters approve the same nearest candidates and ties split the win; Borda loses too (89 → 76), for a reason not yet clear.

## First look: candidates unlike the electorate (2026-10-10)

Run (Claude, small): `sweep.py --one-at-a-time --elections 40 --cand-dispersion 1 0.25 0.5 1.5 --cand-shift 0 0.5 1 2`, ten methods, defaults otherwise (one poll round). 40 elections: point estimates, noisy. Condorcet efficiency with honest ballots (CE), which keeps its meaning across conditions:

| Method | base | dispersion 0.25 | 0.5 | 1.5 | shift 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|
| Plurality | 42 | 29 | 28 | 50 | 62 | 70 | 91 |
| TTR | 55 | 47 | 33 | 82 | 72 | 91 | 98 |
| IRV | 52 | 40 | 40 | 78 | 71 | 88 | 96 |
| Approval | 80 | 41 | 53 | 82 | 72 | 64 | 50 |
| Score | 73 | 50 | 58 | 83 | 75 | 88 | 95 |
| STAR | 80 | 53 | 69 | 95 | 88 | 90 | 98 |
| Borda | 81 | 82 | 83 | 86 | 82 | 82 | 92 |
| Condorcet winner exists | 93% | 85% | 90% | 95% | 100% | 100% | 100% |

- **Clustered candidates (dispersion < 1) hurt as Merrill (1984) found** for plurality, TTR and IRV, and in this model just as much for the rated methods: Approval, Score and STAR lose 20–40 points of CE at dispersion 0.25. Preferences become flat (favourite ÷ least liked 1.5 instead of 4.1), and ballots stretched to each voter's own range (scores, above-mean approval) then exaggerate small differences. Borda is unaffected. Spread-out candidates (1.5) help every method.
- **A shifted field makes elections easier,** not harder: there is always a Condorcet winner, and the candidate nearest the electorate is most voters' favourite, so plurality-type methods find it (CE 91–98 at shift 2). Approval is the exception: nearly everyone approves the same two nearest candidates, and the resulting ties split the win.
- **Caution with VSE and WME across shift values:** when one candidate is nearly everyone's favourite, the random ballot is almost as good as the best candidate, the efficiencies' denominators become small, and every miss looks large (Approval's honest VSE falls to 0 at shift 2 from ties alone). Within one condition the order of methods stays meaningful; across conditions, CE and the raw rates are safer.

## Lock-in: starting from a previous election's result, full strategy (2026-10-10)

Run (user): `--start honest previous --rounds 1 2 3 4 5 6 7 8 --strategic 1 --aggressive 1`, 200 elections, defaults otherwise; Score, Approval, IC-MARS, Minimax, IRV, IRV-3, ER-IRV, TTR, Borda. The first poll shows either the honest ballots or the final ballots of one previous (dummy) election under the same method, its candidates relabelled at random (README, Starting information). Same elections in both; VSE* / WME*:

| Method | start | round 1 | round 2 | round 3 | round 5 | round 8 |
|---|---|---|---|---|---|---|
| TTR | honest | 86 / 84 | 89 / 89 | 89 / 90 | 90 / 93 | 89 / 94 |
| | previous | 40 / 30 | 46 / 34 | 47 / 36 | 47 / 35 | 47 / 35 |
| IRV | honest | 85 / 84 | 85 / 81 | 89 / 84 | 89 / 87 | 89 / 88 |
| | previous | 43 / 33 | 50 / 40 | 54 / 44 | 55 / 44 | 54 / 45 |
| IRV-3 | honest | 88 / 83 | 88 / 84 | 87 / 83 | 87 / 86 | 90 / 90 |
| | previous | 64 / 50 | 72 / 63 | 77 / 68 | 80 / 74 | 80 / 77 |
| ER-IRV | honest | 87 / 85 | 86 / 87 | 88 / 89 | 86 / 88 | 87 / 91 |
| | previous | 75 / 67 | 86 / 82 | 87 / 87 | 88 / 89 | 87 / 91 |
| Approval | honest | 84 / 89 | 88 / 89 | 90 / 89 | 85 / 91 | 90 / 91 |
| | previous | 59 / 44 | 82 / 80 | 84 / 87 | 87 / 86 | 90 / 88 |
| Score | honest | 88 / 92 | 89 / 91 | 91 / 92 | 88 / 90 | 90 / 90 |
| | previous | 82 / 74 | 89 / 81 | 89 / 86 | 88 / 87 | 89 / 90 |
| IC-MARS | honest | 86 / 93 | 89 / 94 | 88 / 93 | 90 / 93 | 90 / 93 |
| | previous | 79 / 72 | 87 / 84 | 86 / 90 | 88 / 91 | 87 / 92 |
| Minimax | honest | 83 / 91 | 83 / 84 | 85 / 91 | 88 / 92 | 85 / 91 |
| | previous | 58 / 44 | 79 / 73 | 79 / 82 | 84 / 88 | 83 / 89 |
| Borda | honest | 50 / 42 | 70 / 65 | 61 / 59 | 68 / 68 | 75 / 70 |
| | previous | 72 / 62 | 63 / 61 | 65 / 60 | 69 / 67 | 67 / 59 |

Second run, same settings (user), more methods; VSE* / WME*:

| Method | start | round 1 | round 2 | round 3 | round 5 | round 8 |
|---|---|---|---|---|---|---|
| Plurality | honest | 78 / 69 | 80 / 72 | 84 / 73 | 85 / 74 | 85 / 74 |
| | previous | 34 / 26 | 36 / 28 | 36 / 29 | 37 / 30 | 37 / 30 |
| Coombs | honest | 80 / 90 | 83 / 88 | 83 / 89 | 81 / 91 | 83 / 91 |
| | previous | 69 / 56 | 76 / 79 | 78 / 84 | 80 / 86 | 84 / 89 |
| BTR-IRV | honest | 77 / 84 | 80 / 83 | 82 / 87 | 86 / 86 | 87 / 87 |
| | previous | 75 / 69 | 82 / 79 | 85 / 81 | 84 / 81 | 88 / 88 |
| MJ | honest | 87 / 90 | 88 / 89 | 87 / 91 | 87 / 94 | 85 / 93 |
| | previous | 69 / 54 | 84 / 83 | 88 / 88 | 86 / 89 | 83 / 92 |
| MCA | honest | 82 / 91 | 88 / 94 | 83 / 91 | 89 / 89 | 88 / 89 |
| | previous | 73 / 62 | 81 / 82 | 85 / 86 | 89 / 89 | 89 / 93 |
| SICT | honest | 80 / 84 | 84 / 84 | 89 / 89 | 86 / 88 | 86 / 90 |
| | previous | 68 / 54 | 83 / 84 | 88 / 84 | 86 / 91 | 84 / 90 |
| STAR | honest | 85 / 87 | 90 / 94 | 87 / 93 | 86 / 92 | 89 / 93 |
| | previous | 80 / 70 | 86 / 86 | 88 / 91 | 88 / 90 | 88 / 93 |
| ARA, Score+, MMPO, MMWV, Minimax-ER, Baldwin-ER | previous, round 1 | 78–86 / 69–81, back within 2–4 rounds | | | | |
| Condorcet-random-drop | both | 81–87 / 77–86 in every round, no difference between the starts | | | | |

- **TTR and IRV stay locked in.** TTR freezes from round 3 on at about 47 / 35, IRV at about 54 / 44, against 89 / 90 with an honest start. The outdated front-runners keep the first choices: everyone marks or ranks first the preferred of the two old leaders, which keeps them in front, and no single group gains by moving to a candidate the polls show as hopeless. TTR's honest runoff only picks the better of the two old leaders.
- **Plurality locks in completely** (34 / 26 → 37 / 30 by round 8, against 85 / 74 with an honest start) and freezes from round 3 on, like TTR. Together with TTR and IRV it forms the group of methods that count first choices only.
- **Every method that reads the whole ballot recovers.** BTR-IRV and Coombs eliminate like IRV, but BTR-IRV decides each elimination head to head between the bottom two, and Coombs eliminates by last places: the strategic ballots still rank the other candidates below the old leaders, and those full rankings carry the information back. Both are back by round 3–8. The same holds for the rated, approval and pairwise methods, the grade methods (MJ, MCA) and SICT; several (ARA, Score+, MMPO, MMWV, Minimax-ER, Baldwin-ER) are hardly affected even in round 1.
- **ER-IRV recovers by round 2–3,** the same split as in the Inverse test: equal first ranks let voters support several candidates at once, so no coordination on one new front-runner is needed.
- **Approval, Score, IC-MARS and Minimax recover** within 2–5 rounds (Approval 59 / 44 → 84 / 87 by round 3; Minimax 58 / 44 → 84 / 88 by round 5). Score and IC-MARS start least damaged (round 1: 82 / 74 and 79 / 72).
- **IRV-3 recovers partly** (64 / 50 → 80 / 77), clearly better than full IRV, for a reason not yet clear. A guess: ballots that rank at most 3 candidates and leave the other old front-runner unranked exhaust in the later eliminations, which weakens the old leaders' transfers.
- **Borda** is unstable under all-aggressive voters whatever the start (honest 50–75 VSE* across rounds), so this test says little about it.
- **The honest start flatters the plurality-based methods.** With honest first polls the popular candidates lead, and TTR and IRV look as good as the rated methods (VSE* about 89). Start them from a different picture and they keep it: their good results depend on the starting information being right. In an electorate whose opinions have moved since the last election, or in which the media picture is wrong, that is the case that matters.

## First look: starting from a previous election's result (2026-10-10)

Run (Claude, small): `sweep.py --start honest previous --rounds 1 3 8 --elections 30 --methods Plurality IRV ER-IRV TTR Approval Score STAR Minimax Borda`, defaults otherwise (strategic share 0.5, aggressive share 0.5). 30 elections: point estimates only, a few points are noise. VSE* / WME*:

| Method | honest start, round 1 | previous, round 1 | honest, round 3 | previous, round 3 | honest, round 8 | previous, round 8 |
|---|---|---|---|---|---|---|
| Plurality | 70 / 54 | 49 / 36 | 63 / 47 | 67 / 51 | 74 / 62 | 66 / 59 |
| IRV | 87 / 73 | 68 / 65 | 90 / 88 | 92 / 85 | 87 / 85 | 90 / 86 |
| TTR | 86 / 85 | 75 / 67 | 88 / 81 | 93 / 85 | 84 / 87 | 87 / 80 |
| ER-IRV | 92 / 73 | 87 / 82 | 87 / 74 | 92 / 85 | 93 / 89 | 95 / 88 |
| Approval | 89 / 91 | 87 / 74 | 92 / 87 | 85 / 82 | 94 / 88 | 94 / 91 |
| Score | 96 / 92 | 93 / 86 | 93 / 91 | 89 / 82 | 90 / 92 | 91 / 90 |
| STAR | 95 / 97 | 95 / 91 | 91 / 92 | 95 / 95 | 91 / 93 | 91 / 96 |
| Minimax | 86 / 97 | 89 / 92 | 88 / 100 | 86 / 99 | 89 / 99 | 89 / 99 |
| Borda | 84 / 90 | 84 / 84 | 83 / 93 | 82 / 83 | 88 / 91 | 84 / 92 |

- **Round 1 pays for the outdated picture,** most under Plurality, IRV and TTR (VSE* 11–21 points lower), less under ER-IRV and the rated, approval and Condorcet methods.
- **By round 3–8 most methods are back,** within the noise of this run; Plurality possibly stays lower (round 8: 66 / 59 against 74 / 62).
- The lock-in is much weaker than with the Inverse-* methods. Two reasons: the relabelled front-runners are random candidates, not the most disliked ones, and kind voters start from their sincere ballots and keep them unless the polls show a gain, which pulls the polls toward the real electorate. A run at full size, and with `--aggressive 1` and `--strategic 1`, would show whether a lasting lock-in remains for the plurality-based methods.

## Model test: Inverse-* methods (2026-10-10)

Run (Claude, small): `sweep.py --strategic 1 --rounds 1 3 8 15 --methods inverse Plurality Approval Minimax --elections 40`, with `--aggressive 0.5` (default) and `--aggressive 1`; defaults otherwise. 40 elections: point estimates only, differences of a few points are noise.

The Inverse-* methods count like their originals, but their sincere ballot expresses reversed preferences, so honest voting elects nearly the worst candidate (VSE about −200). With full strategy and enough rounds, voters should learn to reverse their ballots and get the original's result back.

VSE* / WME* with all strategic voters aggressive (`--aggressive 1`):

| Method | round 1 | round 3 | round 8 | round 15 | original, round 15 |
|---|---|---|---|---|---|
| Inverse-Approval | 5 / −1 | 91 / 75 | 87 / 85 | 81 / 88 | Approval 90 / 85 |
| Inverse-Minimax | −22 / −26 | 76 / 84 | 86 / 83 | 85 / 85 | Minimax 87 / 98 |
| Inverse-Plurality | −46 / −69 | −44 / −67 | −44 / −67 | −44 / −67 | Plurality 88 / 65 |

Second small run (Claude, 20 elections, `--strategic 1 --aggressive 1 --rounds 1 3 8`), the further inverse methods; VSE* / WME* (Inverse-ER-IRV and ER-IRV from a separate run with the same settings and elections):

| Method | round 1 | round 3 | round 8 | original, round 8 |
|---|---|---|---|---|
| Inverse-Score | 51 / 43 | 88 / 93 | 91 / 89 | Score 85 / 91 |
| Inverse-Borda | 40 / 11 | 75 / 77 | 69 / 57 | Borda 89 / 89 |
| Inverse-IRV | −29 / −76 | 21 / −29 | 26 / −11 | IRV 87 / 87 |
| Inverse-ER-IRV | 66 / 16 | 92 / 84 | 91 / 84 | ER-IRV 90 / 83 |
| Inverse-TTR | −24 / −78 | −10 / −69 | 5 / −49 | TTR 97 / 93 |

- **Approval and Minimax: the voters find their way back.** Within 3–8 rounds Inverse-Approval and Inverse-Minimax reach about the VSE* of their originals; Inverse-Minimax stays below Minimax on WME* (85 against 98 in round 15). Sincerity, measured against the reversed preferences the honest ballot expresses, falls to about 37% and 8%: most ballots are reversed.
- **With half the strategic voters kind** (default `--aggressive 0.5`) the recovery is slower and partial (round 12: Inverse-Approval 73 / 70, Inverse-Minimax 66 / 74). Kind voters return to the honest ballot whenever reversing does not strictly pay in the polls, and in a large group a single change rarely does; here the honest ballot is the harmful one.
- **Score recovers like Approval; Borda partly** (20 elections: noisy, and Borda itself is unstable under all-aggressive voters).
- **Inverse-Plurality does not recover.** Honest voting makes two widely disliked candidates the poll leaders. Voters then do the naive leader-rule thing and name the less bad of the two, which keeps both in front: nobody's favourite is a leader, a switch to it gains nothing in the polls, and the electorate stays at about 72% sincerity in a lesser-evil equilibrium. This is the Duverger trap of plurality with a bad start: the strategy model only finds moves that pay against the current leaders, so it cannot coordinate a jump to a new front-runner. Under the ordinary methods voters start from honest ballots, whose leaders are the popular candidates, so this matters less there; but it shows that the poll rounds model level-k responses to the current polls, not ideal strategy.
- **ER-IRV escapes where IRV does not.** Inverse-ER-IRV is back at ER-IRV's level by round 3 (92 / 84), while Inverse-IRV is still at 26 / −11 in round 8. Same count, but equal rankings let a voter put many candidates first at once: the naive exaggeration ties everyone liked at least as much as the preferred front-runner at the top, which hands first-choice support to all of them together, so no coordination on one new front-runner is needed.
- **IRV and TTR lock in too, less completely.** Both decide by first choices first, so the same trap holds: the disliked front-runners of the honest Inverse ballots keep the first-choice votes. TTR's separate runoff, decided by real preferences, picks the less bad of two bad finalists; IRV's eliminations reshuffle a little more. The split is by how the first stage counts: methods that count every candidate on every ballot (scores, approvals, pairwise counts) let voters move many candidates at once and recover; methods that count first choices need coordination on a new front-runner, which poll responses do not produce.

## Current picture: all methods, poll rounds 5, with nomination (2026-10-09)

Run (user): all 59 methods, 400 elections, 99 voters, 6 candidates, defaults (strategic share 0.5, aggressive share 0.5, poll error 2.0x, max score 5), `--rounds 5 --exits 1 --clones 2`. First run with separate runoffs and shared random streams (section below). Replaces the picture of 2026-10-06 further down, which used convergent learning and other axes. Only one round count, so differences between methods not in the "Dominated by" column are point estimates; the frontier's most balanced point is VSE 95.3 / WME 96.7.

Undominated on all four axes (VSE, WME, VSE*, WME*; the analysis since judges dominance on VSE*, WME* and sincerity only, see the note at the top): **STAR, ARA, MARS, Condorcet-random-drop, Minimax, Schulze, Ranked pairs.** Beaten on a single axis only:

| Method | VSE | WME | VSE* | WME* | beaten by (axis) |
|---|---|---|---|---|---|
| STAR | 92.9 | 94.2 | 89.3 | 95.7 | — |
| ARA | 91.5 | 94.9 | 89.6 | 96.2 | — |
| IC-MARS | 91.0 | 95.0 | 89.9 | 96.0 | STAR (VSE) |
| MARS | 90.3 | 95.9 | 88.5 | 97.3 | — |
| Smith//Score | 89.2 | 96.4 | 88.5 | 97.3 | IC-MARS (VSE) |
| MMWVxScore | 88.5 | 96.7 | 88.0 | 97.5 | MARS (VSE) |
| MMWV+Score | 88.5 | 96.5 | 87.0 | 97.0 | MARS (VSE) |
| Condorcet-random-drop | 86.3 | 99.7 | 88.2 | 98.2 | — |
| Minimax | 86.1 | 99.6 | 85.6 | 99.2 | — |
| Ranked pairs | 86.0 | 99.5 | 85.8 | 99.2 | — |
| Schulze | 86.1 | 99.6 | 85.9 | 99.1 | — |
| MMPO, MMWV, MMPOWV, Kemeny-Young, Schulze-ER, Coombs | 85.5–86.1 | 99.1–99.6 | 85.1–86.6 | 98.4–98.9 | Condorcet-random-drop (VSE*) |

- **Two groups along the trade-off, as before:** score-based runoffs and hybrids (VSE 88–93, WME 94–97) and Condorcet methods (VSE 85–86, WME 99–100). Honestly the Condorcet methods sit almost on the frontier's WME end (86.5 / 100); the best hybrids are about 2.5 points inside it on both axes (STAR 92.9 / 94.2 against the balanced point 95.3 / 96.7), between the −2 and −4 guide lines.
- **Strategy mostly moves the hybrids along the trade-off rather than inward:** STAR 92.9 / 94.2 → 89.3 / 95.7, ARA 91.5 / 94.9 → 89.6 / 96.2, MARS 90.3 / 95.9 → 88.5 / 97.3, IC-MARS 91.0 / 95.0 → 89.9 / 96.0. VSE falls 1–4 points, WME rises 1–1.5. Score loses VSE (92.8 → 90.5) and gains no WME (92.3 → 92.5).
- **Condorcet methods lose little WME* under strategy;** they differ in how much. Least loss (WME* 98.8–99.2): Minimax, Ranked pairs, Schulze, Kemeny-Young, Coombs, MMWV, Nanson. More (94.5–97.3): Benham 94.5, Baldwin-Dowdall 95.8, Black-Dowdall 95.9, BTR-IRV 96.5, Black 97.0, SICT 97.0, BTR-Dowdall 97.2, Copeland 97.3.
- **Condorcet-random-drop gains VSE under strategy** (86.3 → 88.2, WME* 98.2) and beats the minimax family, Kemeny-Young and Coombs on VSE*. Its burial share (19%) is high, but the burial does not translate into lost WME.
- **Approval runoff with a separate runoff** is among the leaders under strategy (89.5 / 96.4, from 86.3 / 90.5 honest), close to STAR, ARA and IC-MARS. Honestly it is beaten by the hybrids on VSE and WME. Approval (87.4 / 91.8) and Approval cond. runoff (87.7 / 91.4) gain WME under strategy too, but stay far behind.
- **TTR, Contingent vote and Contingent-ER** end up close together (VSE* 85.9–86.4, WME* 91.4–92.1), with the same honest values. Only TTR has no burial, since its runoff cannot be manipulated; compromise and push-over remain (9.5% and 8.6%). IRV and ER-IRV gain WME under strategy (85.7 → 91.3 and 92.4); IRV-3 less (88.6). Supplementary vote (80.5) and MCA (91.9, VSE* 85.3) trail.
- **Dowdall instead of Borda points does not pay in the Condorcet-type methods.** Baldwin-Dowdall and Nanson-Dowdall are not Condorcet-consistent (CE 96.4 and 92.5 honest; WME 98.1 and 96.2): Baldwin and Nanson elect the Condorcet winner because it always has an above-average Borda score, which Dowdall points do not guarantee. Under strategy the Dowdall versions lose more WME than the originals (Baldwin 97.5 vs Baldwin-Dowdall 95.8, Nanson 98.8 vs 94.3, BTR-Borda 98.3 vs BTR-Dowdall 97.2, Black 97.0 vs Black-Dowdall 95.9) for at most a point of VSE* (Black-Dowdall 86.9 vs 86.2, Nanson-Dowdall 86.4 vs 85.5; Baldwin-Dowdall is lower, 84.2 vs 85.2). Plain Dowdall gains from strategy (85.9 / 84.1 → 87.5 / 88.3) and still trails almost everything, and nomination hits it hard (22% of honest results changed, 0.78 withdrawals per election, against 10% and 0.04 for Borda). An earlier run of the Borda family alone (400 elections, rounds 1–5, nomination on, before the random-stream fix) showed the same pattern in every round. The idea from the poll-rounds section, that Borda-based methods should use Dowdall points, is not supported.
- **Borda** loses most under strategy (86.0 / 94.7 → 85.1 / 91.7; Borda-ER 85.5 / 91.7), then Baldwin-Dowdall (86.0 / 98.1 → 84.2 / 95.8). Minimax, Schulze, Ranked pairs, Kemeny-Young and Baldwin also end up lower on both axes, but by at most 0.5 VSE and 1.4 WME.
- **Nomination** changes the honest result most under plurality-based methods (Plurality 35%, Supplementary vote 32%, TTR and contingent variants 30%, IRV 22%) and approval methods (21%), less under the rated and hybrid methods (8–13%, except MCA, MJ, 321, MDDA, ARA-Top and Reciprocal score at 17–23%) and least under the Condorcet methods (2.5–6%, except Baldwin-Dowdall 7.5% and Nanson-Dowdall 12%). Borda and Approval attract clones (0.65 and 0.81 per election).
- **Sincerity:** rated, approval and hybrid ballots stay at 99.4–99.9 (Reciprocal score 98.7); Condorcet methods at 96–98; Borda is the least sincere serious method (94.3).

## Equal ranking and two-round elections, poll rounds 1–6 (2026-10-09)

Run (user): `--rounds 1 2 3 4 5 6`, 400 elections, nomination on (`--exits 1 --clones 2`), defaults otherwise; the ER pairs plus IRV-3, TTR, Supplementary vote and the three approval methods. Paired differences per round.

- **Equal-ranking versions are mostly ties with their strict versions.** Honest values are identical (sincere ballots are strict). Under strategy, Minimax-ER, Schulze-ER, Baldwin-ER and Ranked Robin vs Ranked Robin-SR are ties in every round. ER-IRV is better than IRV twice (VSE* +1.7 in round 1, WME* +2.9 in round 4) and tied otherwise. Borda-ER vs Borda is a tie in every round, with the sign changing (−1.7 / −2.1 in round 1, +0.8 / +2.1 in round 3, −2.4 / −1.4 in round 6). The behaviour differs consistently: ER voters compromise less and bury more (Borda-ER 7–8% compromise and 24–28% burial vs 13–15% and 19% for Borda). Part of that burial is compromise by tying: a voter who ties the preferred leader with the favourite and moves the other leader down counts as burial, because the favourite stays on top. The prediction that ER versions are always better: equal or slightly better for IRV and the Condorcet methods, not shown for Borda.
- **Approval cond. runoff is equal to or better than Approval:** better on honest WME (+1.5) and, barely, on VSE* in rounds 2 and 4 (+1.4 to +1.6); ties otherwise.
- **Always holding the runoff is better still:** Approval runoff beats Approval cond. runoff on WME* in every round (+3.4 to +6.4), on honest WME (+4.2) and on CE* (+8 to +12); VSE and VSE* are ties. Ending the election at a majority of approvals gives away most of what the runoff adds, since that candidate often is not the head-to-head winner.
- **Borda is the only method that loses clearly from honest to round 1** (86.0 / 94.7 → 83.3 / 88.6; Borda-ER 81.6 / 86.5). The others stay put or gain; TTR gains WME* now that burial is gone (88.5 → 89–92). Borda recovers partly with more rounds (87.2 / 93.9 in round 6).

## Model changes: separate runoffs, shared random streams (2026-10-09)

- **Two-round elections have a separate runoff.** TTR, Approval runoff and Approval cond. runoff used to read the runoff from the same ballot as the first round, so strategic voters could also misrank the finalists, which a real separate runoff gives no reason to do. Now the runoff is decided by sincere preferences and strategy changes only the round-1 marks; TTR also ends in round 1 when a candidate has a majority of the marks (README, Two-round elections). **Every section below this one used the old versions:** what they call TTR is now **Contingent vote** (results unchanged), and the old Approval runoff and Approval cond. runoff no longer exist. The old TTR-ER, which is not an equal-ranking version of a two-round election, is now **Contingent-ER**.
- **Shared random streams.** Strategic nomination and the poll rounds used a random stream per method name; now every method uses the same one (README, Random streams). Pairs that agree on honest ballots used to differ in nomination (e.g. TTR 85.9 / 88.9 vs TTR-ER 85.8 / 87.9, honest, with nomination); they now agree exactly. Results from before differ by sampling noise.

## Strategy model changed: poll rounds (2026-10-08)

**All strategic results above this line (and in the older sections below) used convergent learning:** over 20 rounds, groups of up to 10 voters with the same ranking took turns, each seeing the others' latest strategic ballots, and kept changes that paid. That describes where an electorate ends up after long adjustment, with far more information and coordination than one election offers. It protects methods whose strategy fails only when many use it at once.

The test that showed it (300 electorates, 200 voters, 6 candidates, defaults): when all voters bury at once, each putting the preferred of the two honest frontrunners first and the other last, Borda collapses from VSE 86.9 / WME 93.6 to 46.5 / 29.2 (the dark horse wins), while IRV, TTR, Minimax, Schulze, Baldwin, Ranked Robin and Black keep VSE about 85–86 and WME 97–99. Under convergent learning Borda kept VSE* 87.6 / WME* 94.4 (600 elections, full strategy), because burying groups saw the dark horse rise and backed off; allowing whole ranking groups instead of groups of 10 already lowered it to 81.1 / 86.2 (100 elections).

New default (README, Strategy engine): round 0 is sincere; each poll round publishes only the method's counts from a sample of half the electorate, and all groups (whole ranking groups) answer at once; the strategic axes report round 1. Convergent learning stays available with `--converge 20 --samples 10`.

**First version, and what was wrong with it.** With polls only as uncertain as their sampling error, one round hit Approval and Score hard (40 elections, strategic share 0.5: WME 87 → 69 and 86 → 71). The runner-up's supporters dropped the leader, but the leader's supporters did not drop the runner-up. Two reasons, both about information, not strategy:

- In a 99-voter election the leader's lead (6 votes in the case examined) was far larger than the poll's sampling error (2.8), so the leader's supporters saw a poll in which they were behind in 0.4% of polls: nothing to protect against. The runner-up's ranking groups (8 and 12 voters) could overturn the lead outright.
- Most ranking groups have 1–3 voters. Their change flipped no poll, so their expected gain was exactly 0 and honesty won, although theory (Myerson & Weber 1993) has every voter act on the direction of a positive gain, however small.

**First fix (replaced):** wider poll uncertainty plus a minimum weight for small groups. It helped (leader rule on 68% / 77% of the two sides) but still depended on how close the race was: a leader's supporters protected the lead only when some poll showed it in danger.

**Second fix (replaced): "it helps when it first matters."** Each change was judged at levels with 4^k times wider polls and 4^k times as many voters making it, compared lexicographically. It made the leader rule independent of the gap, but it was slow (26 s per election and core), and changes among hopeless candidates, harmless in the polls as voters see them, got decided by wild scenarios at the outer levels: burial rose to 42–58% for almost every method in a full-strategy run (100 elections), Score included.

**Rule now: leaders and naive exaggerations.** The race that matters is found once per round for everyone: the two candidates that win most often over the polls; if one wins every poll, the polls are widened until another one wins somewhere. Each group then tries a few candidate ballots, the naive exaggerations of its ballot type around those two (approve down to the preferred leader; top score down to it, bottom from the other one, stretched in between; vote for it; move it to the top or the other one to the bottom of a ranking; …), plus its previous answer, and takes the best if it does not lose in the polls: exaggeration wins ties, since strategic voters exaggerate when it cannot hurt, whatever the gap. A few random changes follow, each needing a real gain (README, Strategy engine). About 6.5 s per election and core with all methods.

Approval, same 40 elections, defaults (strategic share 0.5): among strategic voters who honestly approve both poll leaders, 98% of the leader's side and 76% of the runner-up's side approve only the one they prefer after one round, 88% for gaps under 10% of the voters and 87% over; approving only the preferred leader went from 25% of strategic voters (honest) to 78%; Approval VSE* / WME* 91.8 / 89.7 (honest 85.2 / 86.9).

Full strategy, one round, 40 elections (honest VSE / WME → VSE* / WME*, burial, compromise):

| Method | | burial | compromise |
|---|---|---|---|
| Plurality | 60.2 / 52.0 → 82.0 / 57.0 | 0% | 33% |
| IRV | 85.9 / 64.7 → 77.4 / 74.5 | 36% | 22% |
| Borda | 78.9 / 89.1 → 49.0 / 45.7 | 44% | 37% |
| Dowdall | 78.8 / 66.2 → 81.6 / 75.0 | 46% | 27% |
| Minimax, Schulze | 88.5 / 100 → 81–82 / 91 | 41% | 25% |
| Approval | 85.2 / 86.9 → 84.3 / 91.0 | 2% | 0% |
| Score | 93.6 / 85.9 → 88.6 / 94.8 | 3% | 0% |
| STAR | 94.2 / 92.7 → 90.9 / 88.2 | 12% | 1% |
| BTR-score | 92.7 / 95.2 → 88.0 / 94.0 | 10% | 0% |

The burial artefact is gone for rated ballots. Ranked ballots show about 40% burial by design: burying the other leader is the naive exaggeration and is taken whenever it cannot hurt in the polls. That sinks Borda (linear points: the dark horse wins) but not Dowdall (harmonic points: the bottom places hardly differ, and compromise, which helps, dominates); Condorcet methods lose some WME but keep 91. This is the naive first response; how it develops over more rounds is open.

## Aggressive share 0–1 × poll rounds 1–5, full strategy (2026-10-09)

Run (user): `--aggressive 0 0.25 0.5 0.75 1 --rounds 1 2 3 4 5 --strategic 1`, 200 elections per condition, defaults otherwise.

WME* in round 1 by aggressive share (share of strategic voters who exaggerate whenever it cannot hurt; the others only when the polls show a gain):

| Method | 0 (all kind) | 0.25 | 0.5 | 0.75 | 1 (all aggressive) |
|---|---|---|---|---|---|
| MMWV | 91.1 | 92.2 | 94.3 | 89.9 | 79.5 |
| MMPOWV | 93.3 | 93.0 | 93.9 | 89.3 | 84.8 |
| Ranked Robin | 87.3 | 89.2 | 90.7 | 89.9 | 78.6 |
| Schulze | 96.3 | 96.2 | 95.8 | 94.8 | 90.0 |
| Baldwin | 95.9 | 96.7 | 95.7 | 95.3 | 93.9 |
| STAR | 84.9 | 90.9 | 92.9 | 93.6 | 87.4 |
| BTR-score | 90.9 | 96.1 | 96.7 | 95.8 | 90.8 |
| Score | 81.8 | 84.7 | 90.8 | 92.7 | 91.3 |
| Approval | 84.5 | 87.0 | 89.0 | 90.2 | 88.9 |
| IRV | 81.1 | 81.1 | 83.8 | 83.2 | 82.5 |
| Borda | 43.0 | 43.0 | 39.5 | 48.8 | 40.2 |
| Dowdall | 79.7 | 81.9 | 83.2 | 84.7 | 85.1 |

- **The extremes are the outliers.** For nearly every method the mixed settings (0.25–0.75) beat both ends, and between them most methods change little; by round 3 the spread narrows further (MMWV 94–96 for every mixed share).
- **A few kind voters are enough to stop mass burial.** The burial-prone Condorcet methods recover from all-aggressive to three quarters aggressive: MMWV 79.5 → 89.9, Ranked Robin 78.6 → 89.9, MMPOWV 84.8 → 89.3. Blind burial needs nearly everyone to do it.
- **A few aggressive voters help the rated methods.** All-kind is the worst setting for Score (81.8), STAR (84.9), BTR-score (90.9) and Approval (84.5); a quarter aggressive voters lifts them (BTR-score 96.1, STAR 90.9). Aggressive voters protect a lead whatever the gap; with only kind voters the runner-up's side attacks when the polls show a chance and the leader's side does nothing (see the leader rule above).
- **Borda collapses at every share**, all-kind included: its burial pays in the polls, it is not only naive.
- Reading: each extreme is a stress test for one weakness, blind mass burial (all aggressive) or unprotected leads (all kind). Real electorates are mixed, and the conclusions do not hinge on the exact share in between; 0.5 stays the default.

## Poll rounds 1–5, full strategy; aggressive and kind voters (2026-10-09)

Run (user): `--rounds 1 2 3 4 5 --strategic 1`, 200 elections, defaults otherwise (all strategic voters aggressive, the only behaviour at the time).

- **Burial under Condorcet methods is large**, something convergent learning missed: 55–65% of voters for MMPO, MMWV, MMPOWV, Ranked Robin, SICT; about 45% for Schulze, Baldwin, Black. Burying the other leader is the naive exaggeration, taken whenever it cannot hurt in the polls.
- **The methods that held up best before suffer most in round 1:** WME* 79.5 (MMWV), 84.8 (MMPOWV), 78.6 (Ranked Robin), against 90–94 for Baldwin, Schulze and MMPO; STAR 87.4. With more rounds they recover: by round 3, WME* 94.3 (MMWV), 93.6 (MMPOWV), 94.3 (Ranked Robin); STAR 93.6.
- **More rounds help most methods**; the order of methods on WME* changes a lot between rounds (rank correlation with round 1: 0.07–0.26, noise reference 0.75).
- **Borda** collapses in round 1 (WME* 40) and partly recovers (61–68); **Dowdall** gains from strategy (VSE 83 → 88–90, WME 75 → 84–87). Since Borda's weakness under naive strategy comes from its linear points, the Borda-based methods (Black, Baldwin, Nanson, BTR-Borda; Ranked Robin's tiebreak) might do better with Dowdall points; to try once the strategy model is settled.

**Aggressive vs kind voters.** Whether real voters bury when it buys nothing is doubtful, so `--aggressive` sets the share of strategic voters who exaggerate whenever it cannot hurt (aggressive); the others (kind) change their ballot only when the polls show a gain. Default 0.5. A first look (40 elections, full strategy, rounds 1 and 3; efficiencies noisy, burial shares clear):

| Burial | all aggressive | half | all kind |
|---|---|---|---|
| MMWV, MMPOWV | 58–63% | 33–37% | 17–23% |
| Schulze, Baldwin | 39–46% | 22–28% | 13–18% |
| IRV | 38–39% | 20–22% | 4% |
| STAR | 13–16% | 8% | 10–14% |
| Approval | 2–4% | 3% | 2–3% |

What remains with only kind voters is burial that pays in the polls. Kind voters also lose the leader rule's protective half (a leader's supporters see no gain in dropping the runner-up unless a poll shows danger), so under Approval the share mainly decides how strongly leads are protected.

## Defaults changed: width 1.0, 5 clusters (2026-10-08)

Earlier sections used width 1.5 and 3 clusters; from now on the defaults are width 1.0 and 5 clusters (spread stays 0.5). Based on two one-at-a-time sweeps, 400 elections each, seed 1, with exits 1 and clones 2:

- **Width** (0.25–2.0, 5 clusters, 6 candidates). Width only changes how utility falls with distance, not who prefers whom: the Condorcet winner share (96.8%) and the honest WME of ranked methods do not move (MMPO 99.6, IRV 86.0 at both 1.0 and 1.5). It acts on VSE, on rated and approval ballots, and on what strategy is worth. A wide width makes utility nearly linear in distance, so total utility and the majority mostly agree and the VSE axis is compressed: electing the Condorcet winner loses 12.9% of VSE at 1.0 and 8.5% at 1.5, the best-WME end of the frontier is at VSE 86.5 and 91.0, and the honest VSE gap between Score and MMPO is 6.7 and 4.3 points. Favourite ÷ least liked (median voter) is 4.4 at 1.0 and 2.7 at 1.5; very small widths are extreme (378 at 0.25, where electing the Condorcet winner loses 47% of VSE).
- **Clusters** (1, 2, 3, 4, 5, 10; 8 candidates, width 1.5). One cluster leaves little to separate the methods (electing the Condorcet winner loses 1.6% of VSE, frontier best-WME end at VSE 98.3); the tension grows up to 5 clusters (8.2%, VSE 91.4) and eases again at 10 (7.1%, VSE 92.7), where many small clusters blur into a more even electorate. No even/odd pattern beyond noise: the more stable electorate measures move steadily from 3 to 5 clusters (VSE lost 6.6, 7.1, 8.2%; frontier VSE 93.5, 92.3, 91.4), and only the share of Condorcet winners with the highest total utility zigzags (72.7, 75.5, 67.7%), by about the size of its sampling error.

## Score range 0–5, 0–10, 0–100 (2026-10-07)

Settings: the defaults (99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5, 400 elections per condition, seed 1; `--max-score 5 10 100`. Methods: Score, MJ, STAR, BTR-score, MARS, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score, and as fixed references 321 and MCA (three grades, unchanged across conditions, as they should be). The electorates are the same in all three conditions.

Honest VSE / WME → learned VSE* / WME*:

| Method | 0–5 | 0–10 | 0–100 |
|---|---|---|---|
| Score | 96.1 / 93.2 → 91.9 / 91.0 | 96.2 / 93.3 → 92.8 / 90.8 | 96.1 / 93.2 → 93.8 / 91.8 |
| Score+ | 95.6 / 92.6 → 94.4 / 93.3 | 96.1 / 94.1 → 95.0 / 94.3 | 96.2 / 94.2 → 94.3 / 94.1 |
| STAR | 95.4 / 95.6 → 94.8 / 96.0 | 94.9 / 97.4 → 93.8 / 97.0 | 94.6 / 99.0 → 94.3 / 97.2 |
| MARS | 95.8 / 95.6 → 94.3 / 96.4 | 95.8 / 97.1 → 94.2 / 97.1 | 95.0 / 99.3 → 93.9 / 98.1 |
| ARA | 96.0 / 94.4 → 94.2 / 95.7 | 95.3 / 96.6 → 94.7 / 95.9 | 94.3 / 97.7 → 93.8 / 96.6 |
| BTR-score | 94.9 / 95.7 → 93.5 / 97.2 | 94.4 / 98.0 → 93.9 / 98.1 | 92.8 / 99.9 → 92.4 / 99.1 |
| Smith//Score | 95.3 / 95.6 → 94.3 / 97.8 | 94.4 / 98.0 → 94.7 / 98.0 | 93.2 / 99.9 → 92.8 / 98.6 |
| MJ | 93.7 / 81.8 → 92.5 / 88.6 | 92.9 / 80.8 → 92.0 / 87.1 | 92.8 / 80.2 → 93.0 / 84.0 |
| Reciprocal score | 92.0 / 85.5 → 91.6 / 91.0 | 91.3 / 83.3 → 93.6 / 87.7 | 90.5 / 80.2 → 92.9 / 86.8 |
| ARA-Top | 91.4 / 88.1 → 91.1 / 88.4 | 91.1 / 90.0 → 90.3 / 90.6 | 89.1 / 90.5 → 88.5 / 88.0 |

Frontier (the same in all conditions): best WME 100 at VSE 92.6, best VSE 100 at WME 88.4, balanced point 97.9 / 98.0.

- **The WME cap of the rated methods was an artefact of the 0–5 range.** With 0–100, BTR-score and Smith//Score reach 99.9 WME with honest ballots and behave like Condorcet methods on full rankings (VSE 92.8–93.2, the best-WME end of the frontier); STAR and MARS reach 99.0–99.3. The "Rated ballots cap WME" observation below holds for 0–5 only.
- **Score:** honest values do not change at all (rounding to 0–5 already averages out over voters). The learned VSE* rises (91.9 → 92.8 → 93.8), WME* stays about level. Not yet checked with a paired comparison across conditions; and since the score mutation differs above 0–5 (README, Strategy engine), part of the change may come from the learning rather than the range.
- **Hybrids move along the frontier rather than toward it:** a wider range trades VSE for WME (STAR 95.4 / 95.6 → 94.6 / 99.0; MARS 95.8 / 95.6 → 95.0 / 99.3). With honest ballots that brings STAR and MARS close to the frontier at the WME end: the straight line from the best-WME end to the balanced point, which the frontier cannot lie below, gives VSE 94.5 at WME 99.3, and MARS has 95.0. Under strategy they stay well inside it (MARS* 93.9 / 98.1).
- **Score+ gains a little** (WME 92.6 → 94.2), consistent with its re-stretching losing less information on a finer range.
- **Methods that suffer from a finer range:** Reciprocal score (honest WME 85.5 → 80.2), MJ (WME* 88.6 → 84.0) and ARA-Top (VSE 91.4 → 89.1). For ARA-Top this is by construction: on 0–100 practically only the favourite gets the top score, so it approaches a plurality-based approval count. Reciprocal score works with factions of top-rated candidates, which shrink to single favourites as well.

## Number of candidates, 3 to 9 (2026-10-07)

Settings: the defaults (99 voters, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5, 400 elections per condition, seed 1; base 3 candidates, swept over 3–9 (`--cands`).

**Electorates.** More candidates make the task harder for every method: a Condorcet winner exists in 99.8% of elections with 3 candidates and 94.8% with 9; it also has the highest total utility in 88.5% and 68.6% of those; preferences get steeper (favourite ÷ least liked, median voter, 1.7 → 3.0).

**Frontier.** The balanced point stays at about 97 VSE / 98 WME throughout. The best-WME end moves between VSE 91 and 94.5 without a trend; the best-VSE end loses some WME (89.7 → 87.5).

Honest VSE / WME → learned VSE* / WME*:

| Method | 3 candidates | 6 candidates | 9 candidates |
|---|---|---|---|
| Plurality | 75 / 83 → 88 / 91 | 52 / 55 → 82 / 75 | 51 / 46 → 78 / 66 |
| TTR | 93 / 92 → 94 / 98 | 87 / 75 → 90 / 90 | 86 / 70 → 90 / 84 |
| IRV | 93 / 92 → 94 / 99 | 90 / 78 → 92 / 92 | 91 / 72 → 91 / 85 |
| ER-IRV | 93 / 92 → 93 / 98 | 90 / 78 → 91 / 90 | 91 / 72 → 93 / 84 |
| STAR | 95 / 98 → 94 / 99 | 95 / 96 → 95 / 96 | 97 / 91 → 96 / 92 |
| MARS | 95 / 99 → 94 / 98 | | 96 / 92 → 96 / 93 |
| Smith//Score | 95 / 98 → 95 / 99 | | 96 / 92 → 95 / 95 |
| Minimax, Schulze, MMPO | VSE 91–94, WME 100, WME* 99–100 at every count | | |

(Empty cells: not taken from the report.)

- **Plurality-based methods lose most** as candidates are added (vote splitting): honest WME drops from 92 to 70–72 for TTR and IRV, and to 46 for Plurality. Learned ballots recover part of it (voters compromise), but not all.
- **Condorcet methods are very stable:** WME 100 with honest ballots at every count and 99–100 learned; VSE stays in the 91–94 band of the best-WME end of the frontier.
- **Hybrids trade WME for VSE:** STAR, MARS and Smith//Score gain about 2 VSE and lose 6–7 WME from 3 to 9 candidates, consistent with the rated-ballot cap (more candidates share a rating on each ballot, hiding more preferences).
- **ER-IRV against IRV:** ER-IRV is more sincere at every count (e.g. 98.1 against 96.4 with 6 candidates, 98.5 against 96.6 with 9). The strategic axes are mixed: VSE* higher with 9 candidates (93.0 against 91.3), lower with 6 (91.4 against 92.1), and WME* often lower at high counts (85.9 against 89.8 with 8). Equal rankings let voters support a compromise without dropping their favourite, which shows in sincerity; it is not a clear gain in outcomes.
- **Rank stability** against 3 candidates: sincerity stays at the noise reference (0.80–0.92 against 0.81). WME and WME* stay high (0.67–0.94) but mostly below their references (0.95, 0.93), so the order shifts a little, in line with the plurality-based methods falling back. With 3 candidates the methods are so close on VSE and VSE* that the noise reference is low (0.29 and −0.01), so the order there is mostly noise.

## MMPO family under full strategy (2026-10-07)

Settings: 200 voters, 6 candidates, otherwise the defaults; strategic share 1.0, 800 elections, seed 1. Methods MMPO, MMWV, Minimax product, MMPOWV.

- All four are identical with honest ballots (VSE 92.3, WME 100, CE 100).
- Learned: VSE* 91.9–92.2, WME* 99.1 (MMPOWV) to 99.5 (Minimax product), CE* 98.2–98.4, sincerity 99.8 for all; burial 1.7–1.9% of voters, compromise 0.5–0.6%, push-over 0.
- **Every paired difference is a tie** (MMPOWV's WME* is 0.2–0.4 below the others, within noise). The slight tendency for MMPOWV seen in an earlier run did not hold up. MMPOWV replaces Minimax product because it is the simpler form (two factors instead of three) with no measurable loss, not because it is better.

## Reachable frontier and combination methods (2026-10-07)

Settings for both runs: the defaults (99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, 20 rounds), strategic share 0.5 and 1.0, seeds 1 and 2.

```
python sweep.py --strategic 0.5 1 --elections 400 --seeds 1 2 --out combo \
    --methods "Minimax+Approval" "Approval product" "Minimax product" ARA STAR BTR-score Smith//Score Minimax Schulze Score MMPO Minimax-wv
python analyze.py combo-elections.csv --out combo-analysis --compare "Approval product:STAR" \
    "Minimax+Approval:STAR" "Approval product:Schulze" "Minimax product:Minimax"
```

(The command uses the method names of the time; three of these methods no longer exist and Minimax-wv is now MMWV.)

**The frontier.** VSE and WME cannot both reach 100: the candidate with the highest total utility is often not the one with the best worst margin. Knowing every voter's sincere utilities, a rule could at best reach (800 elections per condition):

| Point on the frontier | VSE | WME |
|---|---|---|
| best WME | 92.9 | 100 |
| best VSE | 100 | 89.5 |
| most balanced | 97.6 | 98.2 |

The margin-based Condorcet methods sit at the WME end (about 92.9 / 100 with honest ballots). The hybrids (STAR, BTR-score, MARS, ARA, Smith//Score) sit 2–3 points inside the curve, around 95–96 on both. No method comes near the balanced point; the gap is what the methods lose by seeing ballots instead of utilities, and to strategy.

**Combination methods** (proposed in this lab):

- **Minimax product** (margins × winning votes × pairwise opposition, each as a share 0–1): equal to Minimax with honest ballots, a tie at strategic share 0.5, and at full strategy clearly better than Minimax on WME* (+2.1), CE* (+4.5) and sincerity (+1.7). MMPO, MMWV and Minimax product are within each other's noise (WME* at full strategy 99.0, 98.7, 99.1). The variant without the margin factor, **MMPOWV** (pairwise opposition × winning votes), replaces Minimax product; see the next section.
- **Minimax+Approval** (approvals + worst margin) and **Approval product** (winning votes × opposition × approvals, on ARA ballots): undominated, but inside the hybrid cluster. Minimax+Approval ties STAR on everything except WME* at full strategy (+1.8); Approval product trades 0.7 VSE for 0.8 WME against STAR. Neither moves toward the balanced point of the frontier; both were removed.
- **Under full strategy** the Condorcet methods that measure opposition (MMPO, MMWV, Minimax product) are the most honest of all (sincerity 99.6–99.7 against 98.0–98.5 for the hybrids, burial about 3% of voters against 15–22%).

## Margin band, a negative result (2026-10-07)

Margin band (proposed in this lab, removed again): every candidate whose worst margin is within 5% of the voters of the best worst margin is in a band; the band member with the highest score total wins. Same settings as above, 1200 elections per condition, compared with Score, STAR, BTR-score, Smith//Score, MARS, Minimax, Schulze, Ranked pairs and IRV.

- It tied Smith//Score on every axis at both strategy levels: with a Condorcet winner in 98% of elections, the band and the Smith set nearly always pick the same candidate.
- The hoped-for burial resistance did not appear: 17.5% of voters buried at full strategy, about as many as with the other hybrids, against about 6% with Schulze, Minimax and Ranked pairs.
- **Under full strategy the margin-based Condorcet methods weakly dominate the hybrids:** the hybrids' VSE advantage holds only with honest or partly strategic voters. At full strategy all hybrids drop to VSE* 92.2–92.8, level with Schulze, Ranked pairs and Minimax (92.5–93.1), while the Condorcet methods keep the clearly higher WME* (97–98 against 93–95).
- **Rated ballots cap WME** (on 0–5; with 0–100 the cap disappears, see "Score range"): every method taking pairwise preferences from 0–5 ratings (Smith//Score, BTR-score, STAR, Margin band) reached only about 95 WME with honest ballots. Equal ratings hide preferences, so even a Condorcet rule on ratings misses some sincere Condorcet winners.

## Current picture (6 candidates, Zipf, 2026-10-06)

```
python sweep.py --one-at-a-time --zipf 1 --dims 10 --clusters 3 5 --spread 0.5 0.75 \
    --width 1.5 --cands 6 --group-size 10 1 0 --seeds 1 2 --elections 400 --out model \
    --strategic 0.5 1 --rounds 20 --exclude "Condorcet-random-drop" "Reciprocal score"
python analyze.py model-elections.csv --out model-analysis
```

Base: 99 voters, 6 candidates, 10 Zipf dimensions, 3 clusters, spread 0.5, width 1.5, group size 10, strategic share 0.5, 20 rounds; 800 elections per condition. These are now the defaults.

Methods that no other method clearly beats on PE, WM, PE* and WM* (out of 6 conditions):

| Method | Undominated in |
|---|---|
| Schulze, MARS | 6 |
| Minimax, Ranked pairs, BTR-score | 5 |
| STAR | 4 |
| all others | never |

- **Two groups lead:** margin-based Condorcet methods (Schulze, Ranked pairs, Minimax) and score runoffs (MARS, BTR-score, STAR). Score runoffs are ahead on PE, the Condorcet methods on WM.
- **Burial sensitivity separates the Condorcet methods.** Their honest values are identical, but Benham, BTR-IRV, SICT and Copeland lose 1–2 points more WM under full strategy than Schulze, Ranked pairs and Minimax.
- **Plurality-based methods gain from strategy, but never enough.** Under full strategy TTR's WM rises from 6.5 (honest) to 13.7 and IRV's from 7.3 to 13.2, while Score drops from 13.0 to 10.9; the margin-based Condorcet methods lose least (15.7 → 14.5–14.9).
- **What changes the method order:** full strategy (PE* rank agreement 0.46 against 0.79 from noise), 5 clusters (0.56) and group size 1 (0.58). Spread 0.75 does not.
- **Cycles stay rare** with 6 candidates too (1.6–3% of elections), so how a method resolves cycles matters less here than how well it resists burial.
- **Group size changes little with 6 candidates,** partly for a mechanical reason: groups are voters with the same full ranking, and with 720 possible rankings most groups are small whatever the cap (see README, Coordination).

The same kind of sweep with 4 candidates (width 1.0 1.5 2.0, spread 0.5 0.75 1.0, clusters 3 2 5, strategic share 0.5, 800 elections per condition) gave a similar but wider set: Schulze (undominated in 6 of 7 conditions), STAR, MARS, BTR-Borda (5), Minimax (4), BTR-score, SICT, Baldwin, Approval runoff (2), ARA, Score (1). Width hardly changed the order there; spread and 2 clusters did, mainly under strategy.

## Voter model (2026-10-05/06)

**Plain dimensions, one parameter at a time** (99 voters, 4 candidates, group size 10, strategic share 0.5, 15 rounds, 200 elections per condition): most apparent reordering was within sampling noise, especially on PE*. The electorates themselves (1500–2000 generated per setting) explained most of the rest:

| Setting | Condorcet winner also highest total utility | PE lost electing it | Favourite ÷ least liked (median) |
|---|---|---|---|
| base (2 dims, width 1) | 80% | 7.7% | 3.9 |
| dims 4 | 76% | 10.9% | 6.0 |
| width 0.5 | 74% | — | 17 |
| width 2.0 | 85% | — | 2.0 |

(The PE-lost figure is a ratio of sums, as in the report.) More dimensions and narrower width make the Condorcet winner less often the best candidate by total utility, which lowers PE for Condorcet and plurality-based methods and favours rated methods. Width 0.5 makes preferences implausibly steep (a voter values the favourite 17 times the least liked candidate).

**Zipf dimensions** (dimension i with standard deviation 1/i, after Clay Smith) make the number of dimensions almost irrelevant: from about 4 dimensions on the electorates are the same within noise.

| Model (width 1) | Condorcet winner also highest total utility | PE lost electing it | Favourite ÷ least liked |
|---|---|---|---|
| plain, 2 dims | 80% | 7.7% | 3.9 |
| Zipf, 4 dims | 80% | 7.0% | 2.9 |
| Zipf, 10 dims | 77% | 8.7% | 2.8 |
| Zipf, 20 dims | 79% | 8.5% | 2.8 |

In a direct comparison (200 elections each), Zipf kept the method order on WM and on PE at width 1, widened the PE gap between rated and Condorcet methods by 1–2 points, and hit Plurality hardest (PE 81 → 71). With Zipf the methods cluster more clearly by type.

## Approval with conditional runoff (2026-10-05)

In Germany a winner is expected to have more than half the votes. The tempting chain "use approval instead of TTR, and add a runoff only when nobody has a majority" does not work in the model.

Settings: 600 elections, 200 voters, 6 candidates, plain 2 dimensions, 3 clusters, spread 0.5, width 1.0, group size 10, full strategy, 20 rounds (CE was still an axis then). Paired differences, percentage points (ties in brackets are within noise):

| Pair | PE | CE | PE* | CE* | dissonance |
|---|---|---|---|---|---|
| ACR − Approval | tie | tie | tie | tie | +2.8 |
| Approval runoff − Approval | +2.0 | +19.2 | tie | +9.4 | +1.4 |
| Approval runoff − TTR | +3.3 | +20.4 | +1.5 | tie | −1.3 |
| ACR − Approval runoff | −2.2 | −19.2 | −1.6 | −9.0 | +1.4 |
| ACR − TTR | tie | tie | tie | −7.5 | tie |

("Dissonance" was the complement of today's sincerity.)

- With honest ballots the most approved candidate had more than half the approvals in 97% of elections, and the top two both did in 84%. So the conditional runoff almost never happens: ACR is Approval, plus a threshold that invites strategy.
- A majority of first choices implies a majority against every rival; a majority of approvals does not. The 50% rule checks for a real majority in plurality, but almost nothing in approval.
- Approval with an obligatory runoff is better than TTR on three of four axes, ties on CE*, and has less dissonance. If a 50%-winner is required, that is the version to use.
- Conditional TTR needs no separate test: in the model it is identical to TTR (README, Strategy engine).

## Coordination: burial against Condorcet methods (2026-10-03)

Plain 2 dimensions, 99 voters, 4 candidates, full strategy, 60 elections; CE with learned ballots:

| Method | Whole ranking groups | Every voter alone | Burial (groups / alone) |
|---|---|---|---|
| TTR | 94.9 | 94.1 | 0.1% / 0.3% |
| IRV | 96.6 | 91.5 | 1.8% / 1.0% |
| Copeland | 83.6 | 100.0 | 20.1% / 2.0% |
| Minimax | 89.8 | 100.0 | 20.6% / 1.3% |
| BTR-Borda | 91.5 | 100.0 | 23.2% / 1.2% |
| BTR-score | 89.8 | 96.6 | 20.5% / 2.8% |
| STAR | 89.8 | 98.3 | 18.6% / 3.1% |

Burial needs blocs; compromise works for individuals. With organised factions, Condorcet methods and score runoffs lose much of their Condorcet efficiency to burial, while TTR and IRV are carried toward the Condorcet winner by compromise. With individual voters the picture reverses. This is why every strategic result must be read with its group size.

## Strategy engine checks (2026-10-03/04)

- **Rounds:** with 50 voters and 6 candidates, PE*, CE* and sincerity changed only within noise between 20 and 40 rounds; at 10 rounds they were still moving. 20 rounds is the default.
- **No early stop:** stopping once a round passes without any accepted change was tested (200 elections, 99 voters, 4 candidates, plain 2 dimensions, whole ranking groups, full strategy). It saves most of the run time, but quiet rounds are common long before learning has finished. Stopping after one quiet round changed 55 of 99 method × axis values clearly (sincerity higher for almost every method, CE* up to 9 points higher for Condorcet methods); after two quiet rounds still 50.
- **Honesty wins ties:** without it, deviations that stopped paying off stayed, and dissonance grew with the number of rounds.
