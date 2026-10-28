# Campaigns and risk: a design

Status: implemented as an experiment (2026-10-27): `--campaign` and `--risk` in run.py and sweep.py, both off by default; first results under First trials, below. Changes from the design as first written, after the first trials: only bold voters follow recommendations, a campaign's move is one simple move (no exaggerations), and a voter listens to one campaign only, that of the candidate they like most among those campaigning (see The moves and The voter's choice). It extends the poll-rounds strategy model ([model.md](model.md), Strategy engine) by two optional parts, both off by default: campaigns, which give voters coordinated moves to choose from, and a risk attitude per voter, which replaces the fixed split into aggressive and kind voters. With both off the lab gives exactly the results it gives now.

## The problem

The strategy model finds moves that pay for one group of voters on its own: a group of voters with the same ranking judges a ballot by its own expected gain over the polls. Moves that pay only if many voters make them together are not found. Two cases from the runs:

- **CL else Score winner** (a reference method) elects the Condorcet loser of the ballots whenever there is one, at the defaults in nearly every election (VSE\* about −213). To escape, one of the loser's pairwise defeats has to become a tie: a median of 21 votes out of 99, so about 11 voters reversing one pair. Ranking groups have a median of one voter. No group is ever decisive, so none moves, and sincerity stays at 99%. The useful move is also not among the moves tried: it raises the voters' most disliked candidate above a third one, while the naive exaggerations work around the two poll leaders.
- **Inverse-Plurality** stays in a lesser-evil equilibrium (journal, Inverse methods), the Duverger trap: a switch to a new front-runner pays only if enough others switch too.

Real voters face these problems too, and campaigns are how they sometimes solve them: a party or a candidate tells its supporters what to do. Whether enough of them follow decides whether it works. The design should give voters such recommendations without making them follow, so that coordination succeeds where enough voters gain from it and fails where it is too hard.

## The rule in three parts

1. **Every candidate campaigns, every round, at the same time.** A campaign is one public recommendation: a move each voter can apply to their own ballot. A candidate chooses the move that would do most for them, by their own preferences, if a share φ of their audience followed it. The **audience** of a candidate is the voters who prefer them to their strongest rival in the polls: the poll winner, or for the poll winner, the runner-up. A candidate recommends nothing if no move would help them.
2. **Voters judge every option by their own gain in the polls**, as now. A recommendation is one more option, next to their previous ballot, the naive exaggerations and random search. What makes it different is what it promises: a voter follows a recommendation if it costs them little on their own and would help them if the audience followed.
3. **A voter's satisfaction decides how much risk they take.** Voters who expect a good result are careful; voters who expect a bad one are bold, up to accepting moves that lose a little in the polls.

Everything else should follow from these: coordination that succeeds or fails, defence by the leader, counter-campaigns, alliances between candidates.

## Part 1: campaigns

### The moves

A recommendation is a pair (X, Y): "vote as if you liked X a little more than Y". Every follower applies it to their own preferences: a follower who already prefers X to Y votes their sincere ballot; one who prefers Y to X moves X up to just above Y in their own preferences, and the ballot type's sincere ballot does the rest. Every ballot type has a sincere ballot, so campaigns need no new ballot code. For the Inverse-\* methods the ballot is read the ordinary way, as the naive exaggerations are, so following a campaign does not bring back their reversed honest ballot.

Example, score ballots 0–5, a follower with utilities A 1.0, B 0.7, X 0.3, W 0.0, and the campaign "W above X" (W the Condorcet loser who is winning). The follower prefers X to W, so they move W to just above X: the sincere ballot is A 5, B 3, X 0, W 0, W and X tied at the bottom, half a vote less for X over W. On a ranking, W moves to just above X. A follower for whom Y is near the top lifts X a long way, which sincerity counts.

**Why only this one move.** The first version also offered the naive exaggerations for X against Y (top score down to X, nothing from Y down). With them, campaigns cost most methods 5 to 20 points of VSE\* with all voters strategic (First trials, below): every candidate found an exaggeration that would work for it, and bold voters followed. With the simple move alone, campaigns are much weaker and much cheaper.

The pairs considered are those with the poll winner W in them: (X, W) and (W, Y) for every other candidate, so 2(C − 1) moves. This keeps the cost linear in the number of candidates. It covers:

- promoting oneself or an ally against the winner: (X, W);
- the winner's defence: (W, runner-up);
- moving the winner up against a third candidate, which is the escape from CL else Score winner: (W, Y) with W the Condorcet loser.

It leaves out campaigns about two candidates who are both not winning, for example a push-over in IRV that raises a weak candidate above the runner-up to change the elimination order. That restriction is a choice for cost; see Open points.

### The candidate's plan

Each candidate c, in each round, for each move o:

- **Without the campaign:** the polls of this round, everyone voting as in the last round.
- **With the campaign:** the same, with a share φ of the audience casting the move instead of their last ballot (those already following it change nothing).
- **Value of the move to c:** c's expected utility over the polls with the campaign minus without. Candidates have positions, so they have utilities for every candidate (the same utility function as the voters, with themselves at the top); with strategic nomination these are the existing candidate utilities (`cu` in nomination.py), and a clone campaigns with its original's preferences.

c recommends the move with the highest value, if that value is positive.

**φ, the safety margin.** The candidate plans with φ of its audience, not all of it, so that the move still works if part of the audience does not follow. The candidate does not know who is strategic: non-strategic voters never follow, and strategic ones only if it pays them (part 2). So φ is the candidate's expectation, and when it is wrong, the campaign fails. One parameter, `--campaign-share` (default 0.5: the safer assumption, a campaign that works with half its audience; a sweep shows how much it matters).

### The voter's choice

Only bold voters (aggressive ones, with the fixed split) get recommendations, and each group listens to one campaign: that of the candidate it likes most among those campaigning (loyalty; with alliances that can be a campaign for another candidate). For that recommendation:

- **g1, the cost alone:** the group's expected gain in the polls from casting the move instead of its sincere ballot, everyone else as they are (as for any option now).
- **g∗, the promise:** the group's expected gain if the campaign succeeds as planned: its utility with the campaign minus without, from the polls the candidate computed (no new count needed).

A bold group follows it if g∗ > 0 (it would help them) and g1 ≥ −tolerance (it does not cost them too much on their own; the tolerance comes from part 2). Among all options the group takes the one with the highest g1; ties go to the higher g∗. Other options (exaggerations, search) have g∗ = g1, so a recommendation wins over a naive exaggeration that gains as little alone only if it promises more. Careful voters act as before: they take a move only if it strictly gains them something alone, so they join a campaign once the polls show it working.

As first designed, careful voters also followed costless recommendations, and every voter chose among all campaigns the one promising most. In first trials that made a fifth of all voters follow some campaign in every election, the rounds stopped settling, and the ordinary methods lost several points: almost every move is costless for one small group alone, so "costless and promising" let nearly everyone follow whichever campaign promised them most.

What this does:

- **A campaign that costs nothing is followed by the bold voters it helps** (g1 = 0, g∗ > 0). Raising the Condorcet loser above a third candidate changes nothing for one group alone, so it costs nothing; if enough bold voters follow, the next poll shows the loser's defeat gone and the Score winner ahead, and careful voters may join where it now pays them.
- **An equilibrium, not a switch.** Once a campaign works, more followers would add nothing (g∗ falls to 0), and careful voters, for whom following is not worth it any more, return to their own ballot; bold voters keep following as long as it does not cost them more than their tolerance (strategy wins ties for them). Followers fall away until the outcome is at risk again, the polls show it close, following pays again, and some return. So support settles where the campaign is just about enough, and whether that is enough to win depends on how many voters are willing to strategize and to accept a cost: on the voters' values, not on a rule. If it does not settle but swings from round to round (all groups answer the same poll at once), the share of settled elections shows it; then a partial adjustment (only some groups answer each round) would be the next step, not a fix by hand.
- **A campaign that costs something needs bold voters too.** In Plurality, switching from the lesser evil to a third candidate loses in the polls (the lesser evil may lose), so careful voters stay; only voters bold enough to accept that loss follow. If they are enough, the next poll shows the third candidate gaining, the cost falls, and careful voters join. If they are too few, it fails. That is the Duverger trap, kept as a real problem with a real way out.
- **Defence and counter-campaigns** come from the same rule: the poll winner campaigns too, for its own audience.
- **Alliances**: a candidate who prefers B to the current winner may recommend (B, W), and B's supporters may follow A's recommendation. This is the first step towards asset voting.
- **Cycles**: in a Condorcet cycle all candidates campaign in the same round, so none of them gets the first move. Whether the result settles or keeps turning depends on the rounds; the rounds sweep shows it.

## Part 2: risk

### Satisfaction

Per voter and round, from the polls of the round (before anyone answers):

satisfaction = (expected utility of the poll winner − least liked) / (favourite − least liked)

on the voter's own scale, 0..1, as the stake of adaptive voters is. It is per voter, not per group: two voters with the same ranking can care very differently.

### Attitude

Each voter draws a personal risk threshold once (uniform on 0..1, its own random stream, the same for every method, as the adaptive thresholds). In each round:

- **careful** if satisfaction ≥ threshold: takes a move only if it strictly gains (today's kind voters), except costless campaigns that would help them (part 1);
- **bold** if satisfaction < threshold: takes a move if it does not lose in the polls (today's aggressive voters), and accepts a loss up to its tolerance:

  tolerance = κ × (threshold − satisfaction) × (favourite − least liked)

  so a voter far below their threshold accepts more. κ is the one parameter (`--risk-tolerance`; κ = 0 gives exactly today's aggressive voters).

Groups are voters with the same ranking and the same attitude, rebuilt each round (as for adaptive voters); a group's tolerance is the sum of its members'. Two attitudes, not a continuum, because groups act as one bloc: splitting every ranking by a continuous value would make all groups single voters, which alone would change the results.

`--risk` replaces the fixed aggressive share (`--aggressive`); the fixed share stays the default and can be compared with it, as the fixed strategic share is with adaptive voters.

### Where the tolerance applies

To the candidate options (the previous ballot, exaggerations, recommendations): the group takes the best of them if its g1 is at least −tolerance. Random search stays as it is (a proposal is kept only if it strictly gains), so that bold voters do not wander at random; the risk they take is in following moves that do not pay yet, not in noise. Without campaigns, bold voters therefore accept small losses for naive exaggerations, which is a small change; with campaigns, they are the ones who start a coordination that careful voters complete.

### Satisfaction as a reference point

The scale runs from the least liked to the favourite. A voter whose favourite is a fringe candidate is then never fully satisfied and is more often bold, which is plausible (such voters do vote expressively more often) but affects every method. The alternative, the better of the two poll leaders as the top, measures "worse than what seemed possible", but misses the Duverger trap: there the lesser evil does win. Proposed: the favourite; the checks below look for unfair advantages for fringe voters.

## What it costs

Each candidate's plan is a search like one voter group's: 2(C − 1) moves (10 at the defaults), each counted over the polls for all candidates at once. A group counts only the one recommendation it listens to, and only if it promises something. Measured: about 1.25 times the run time with campaigns on (the first version with exaggerations took about 3 times). No random numbers are drawn for campaigns, so the random streams of the learning stay as they are; methods without feature sums (Geometric median, Condorcet-random-drop) apply a move to the first ⌈φ × size⌉ voters of the audience.

## Outputs

New run parameters (`campaign`, `campaign_share`, `risk`, `risk_tolerance`; older files read as off) and new columns per learned row: the share of voters following a recommendation, the share of strategic voters who are bold, and the candidate whose campaign has the most followers. The report adds them to the strategy section. Fingerprints and the regression check run with both parts off, so they stay as they are; a test checks that off gives exactly today's results.

## Checks

Unit tests:

- the move puts X above Y (scores: at least level), keeps the runoff ranking of two-round ballots sincere, and is read the ordinary way for the Inverse-\* methods;
- a small profile where the Score winner's campaign (W, Y) removes the Condorcet loser, and the campaign is recommended and followed;
- a profile where a campaign costs its followers something, followed with bold voters and not with careful ones;
- off: the same results as without the code (regression check, all equal).

Predictions to test with short runs, then large ones:

1. **CL else Score winner and the inverse methods recover** with campaigns; Inverse-Plurality partly, and more with `--risk`.
2. **Methods that need no coordination change little**: Score, STAR, Minimax, Ranked pairs within a point or two of VSE\* and WME\*.
3. **Plurality**: the Duverger trap opens sometimes, more with bold voters; an effect to report, not to avoid.
4. **No hidden advantage**: the gain from campaigns and risk per kind of voter (central against fringe favourites), and the strategy types (compromise, burial, push-over) by attitude, do not show one side winning by construction.
5. **Rounds**: the result settles within a few rounds, or the share of settled elections shows that it does not; cycles in particular.
6. **Parameters**: φ (0.5, 0.75, 1) and κ (0, 0.1, 0.3) swept, to see how much the results depend on them.

## First trials (2026-10-27)

**First version** (moves with exaggerations, every voter choosing the most promising campaign). Sweep: 200 elections at the defaults (adaptive voters). It lifted the extreme methods (CL else Score winner +85 VSE\* with risk), but the Condorcet methods lost 2 to 5 points of WME\*, and the rounds rarely settled. A run of 200 elections with all voters strategic and φ = 1 (`--strategic 1 --adaptive 0 --campaign 0 1`) showed losses of 5 to 10 points of WME\* for most methods. Short runs with all voters strategic (60 elections) traced it to the exaggerations: Score fell from 87 to 67 VSE\*, IRV from 84 to 68; with the simple move alone the ordinary methods stayed within a few points.

**Basic version** (the simple move, loyalty, only bold voters follow). Sweep: 200 elections, all voters strategic (`--strategic 1 --adaptive 0`), the standard set and the extreme methods, `--campaign 0 0.5` × `--risk 0 1`; paired differences against both off, VSE\* / WME\*, percentage points; "clear" when the 95% bootstrap interval excludes 0.

| | campaigns only | campaigns and risk |
|---|---|---|
| CL else Score winner (−200 / −134 without) | +40 / +20 | +56 / +26 |
| Inverse-Plurality | +38 / +33 | +69 / +52 |
| Inverse-Score, Inverse-Borda, Inverse-Minimax | (moves read reversed in this run; fixed since, see below) | |
| Condorcet methods | VSE\* +2 to +6 (Ranked pairs, Kemeny-Young clear), WME\* within ±2 | VSE\* up to +6 (Ranked pairs, MMPO clear), WME\* within ±1.5 |
| Plurality | −9.5 / −4.9, clear | −10.9 / −4.6, clear |
| IRV family, MJ, Score, MARS, 321, Approval | mostly within ±5; clear: IRV −4.9 VSE\*, MJ −4.6 VSE\*, Score −3.6 WME\*, MARS −3.2 WME\* | mostly within ±5; clear: Approval −7.1 / −4.4, Score −6.0 WME\*, 321 −6.7 WME\*, MJ −5.7 / −4.3, MARS −5.1 VSE\* |
| STAR, Score runoff, BTR-score, Smith//Score, ARA, Approval runoff | within ±2.5, not clear (Approval runoff −3.8 WME\* clear without risk) | within ±3, not clear |
| Settled | about 0.5 → 0.35 | about 0.5 → 0.2 |
| Voters following a campaign | 7–22% (28% for the extremes) | 6–18% (43–55% for the extremes) |

The first run of the basic version took the follower's ballot for the Inverse-\* methods from their honest (reversed) ballot, which made following a campaign undo the voters' own reversal (Inverse-Score −94 VSE\*). Read the ordinary way (as now), 60 elections with risk on: Inverse-Score −25 → +54, Inverse-Borda −33 → +2, Inverse-Minimax −7 → +39, Inverse-Plurality −89 → −40 VSE\*.

- **The extremes recover in part**, more with risk on: the bold voters are the dissatisfied ones there. All stay below the random ballot except the inverse methods on scores and rankings.
- **The ordinary methods move by a few points**, both ways: the Condorcet methods gain a little VSE\*, several rated and median methods lose a little; most differences are not clear at 200 elections.
- **Plurality loses about 10 points of VSE\***, clear. Coordination is what Plurality is most sensitive to, so this may be a real effect (campaigns concentrate votes on fewer candidates) rather than an artefact; it needs a closer look.
- **The rounds settle less often**, still.

**Swings between rounds** (2026-10-28; `sweep.py --methods extended references --elections 500 --campaign 0 0.25 --rounds 1 2 3 4 5 6`; per election, how often the winner changes from one round to the next, and how often it returns to the winner of two rounds before, over rounds 3–6). For the ordinary methods campaigns add a few points of changes (Plurality 10.2% → 14.4%, TTR 9.6% → 13.2%, IRV to 13.6%), with returns about a third of the changes as without campaigns: no swing. CL else Score winner does swing: changes 12% → 35%, returns 4% → 15%. A campaign removes the Condorcet loser's defeat, the Score winner leads the next poll, the followers stop following (adaptive voters in part because the race no longer matters to them), and the loser is back: a successful campaign undermines its own motivation. Left as it is; partial adjustment would damp it, but it would change every method and add a parameter for one test method. (Lottery results, as of Random pair runoff, Anti-plurality and TRASH, cannot be compared this way and are left out.)

## Open points

1. **φ**: 0.5 suggested; whether it should be one number or depend on the size of the audience.
2. **Pairs with the poll winner only**: cost against completeness (push-over campaigns about two non-winners are left out).
3. **Careful voters following costless campaigns**: dropped after the first trials (see The voter's choice); every campaign now depends on bold voters to start it.
4. **Stability**: the rounds settle less often with campaigns (see First trials); partial adjustment (only some groups answer each round) is a candidate.
8. **Plurality**: whether its loss with campaigns is a real coordination effect.
5. **Satisfaction's reference**: the favourite (as designed) or the better leader.
6. **κ and the threshold distribution**: uniform thresholds as for adaptive voters; whether the risk threshold should be independent of the adaptive threshold (as designed: separate draws) or the same draw.
7. **Convergent learning**: campaigns only in poll rounds; `--converge` with `--campaign` is refused.
