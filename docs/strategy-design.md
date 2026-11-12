# Strategy: Design Note for the Next Engine

Status: a proposal under discussion (2026-11-10, revised 2026-11-11 and 2026-11-12), implemented as an experimental engine (`--engine inertia`, strategy/inertia.py; measurements below). The model below is the revision of 2026-11-12; the first version (rules with cutoffs) is summarised under What Changed. The existing engines stay for comparison. The engine in use is described in [model.md](model.md) (Individual voters); this note says what is wrong with it, proposes one principle to replace its separate motivations, and lists what is still open.

## Why a New Design

The current engine (pooled discovery, `--engine pooled`, versions 2.x and 3.0) values a move as if a fixed share of a voter's type made it (`--coordination`, 0.25), and each voter then decides alone, by the move's value against a personal cost of voting insincerely. Three problems:

1. **The assumed share is a free parameter, and it can be wrong.** With adaptive voters, the share of a type that actually follows a move is 0.17–0.32 (median, by method; 100 elections), close to 0.25. With all voters always strategic it is 1.0, four times the assumed share. Valued consistently (`--coordination 1`), that condition gives different results (300 elections, VSE\* / WME\*):

   | Method | Assumed share 0.25 | Assumed share 1 |
   |---|---|---|
   | Schulze | 85.8 / 97.6 | 82.5 / 91.3 |
   | Soft Minimax | 84.6 / 96.1 | 86.2 / 97.6 |
   | IRV | 71.3 / 77.8 | 86.7 / 94.5 |
   | Plurality | 74.0 / 73.5 | 78.2 / 79.5 |

   So the statement of the 2.1.0 findings that the Condorcet methods hold best under full strategy with pooled discovery came from the mismatch, as did the collapse of IRV there.
2. **Strategy depends on coordination in an unrealistic way.** At `--coordination 0` nobody votes strategically, and the amount of strategy follows a parameter no voter could know.
3. **There is no single reason behind the rules.** The engine grew from several motivations (shared discovery, a cost of insincerity, trust in moves). Whether voters act boldly or cautiously, and how naive strategy backfires, have no place in it; the aggressive and kind voters of version 1.0.0 were a fixed trait, set in advance.

## The Principle

A single vote never decides an election, so a voter who weighed their own chance of being pivotal would never vote strategically, or vote at all. Real voters reason about a group: "a vote for X is wasted", "we have to stop Y". The question they answer is *what should people like me do?*, and they do their part if it is worth the effort of changing how they vote. Most voters know the polls but do not react to them: they vote as they always meant to, unless the polls show things going badly enough.

> A voter asks what people like them could do in the races that matter, guessing how many of them will do it, and leaves their honest ballot only for a move that beats it by more than their inertia. They reason from their honest ballot every round.

The first half is close to the rule-utilitarian (or ethical) voter of the literature on turnout (Feddersen and Sandroni 2006; Coate and Conlin 2004), applied to the choice of ballot, with one difference: the voter does not know how many will follow and has to guess. That application is this note's own step. The second half is inertia, or satisficing (Simon 1955): a voter keeps their own position until something clearly better is on offer. The poll rounds then do the work of reasoning about the others: each poll shows what the others did in answer to the last one, and voters respond to it as far as it moves them.

## The Model

The model has four parts: who is on a voter's side (sides), how a voter values a move without knowing how many will make it (the guess), what they remember from earlier rounds (memory), and when they leave their honest ballot (inertia). Voters differ in **one trait only**, their inertia.

### Sides: one per race

The polls give the candidates likely to win (the top three, as now). Each pair of them is a race, and in each race a voter has a side: everyone who prefers the same candidate of the pair. A voter with C > B > A is on B's side against A, on C's side against B and on C's side against A. With three places there are three races and six sides, and every voter is on three of them.

A side is defined by one preference only, as real camps are: "everyone who wants to stop A" includes voters who disagree about everything else.

*Alternative:* the six types of the current engine (the voters with the same order of the top three), each with one rule. Simpler, but it splits a camp into groups that cannot see their joint strength. The per-race sides are the proposal; the types remain the fallback if the sides prove too costly or unstable.

### Inertia: unwillingness to leave home

Each voter has an inertia θ, a share of their own utility range, drawn per election from their own random stream (the same for every method). Their honest ballot is their home. Every round they compare the best move available to them (below) with their honest ballot, and cast the move only if it beats the honest ballot by more than θ; otherwise they vote honestly. Going back home costs nothing.

- **Low inertia:** makes a move for small advantages.
- **High inertia:** stays honest unless the polls show things going badly enough that a move gains a lot. A poll much worse than expected shows up as a large gain from a move, so no separate notion of what each voter expected is needed.
- **Always strategic voters** are those with θ = 0; voters outside the strategic share never leave home.

The reference is the honest ballot, not the ballot cast last. A voter who reasoned from their last strategic ballot would forget where they started: a move that no longer helps but does no harm would never be taken back, and strategy would grow from round to round (see What Changed). Reasoning from home means a move is kept only as long as it keeps paying more than the voter's inertia.

Inertia is smooth at the level of the electorate: each voter has a sharp threshold, but with θ drawn from a continuous distribution, the share who make a move grows smoothly with the gain it offers.

### The guess: an unknown share of the side

For each side the obvious moves for its candidate against the other are tried (the naive exaggerations and the push-over), each voter casting the move from their own utilities. A voter does not know how many on their side will make a move, and does not count the others' decisions; they guess.

- The move is valued against the side at home: the round's polls counted again with the side's members on their honest ballots, and with a share s of them casting the move instead. Everyone else is taken to vote as the poll shows; what the other sides actually did is in the poll.
- **Round 1:** the voter has no idea how many will follow, so the value is averaged over all shares from 0 to 1 (evaluated at the midpoints of four equal parts). A move that works only if nearly the whole side joins is worth little; one that works with a few is worth nearly its full gain. The guess has no parameter, and it does not assume that the side coordinates.
- **Later rounds:** after a round in which the move was weighed, the voters see what share of their side cast it, and guess around that. The shares are the quartile midpoints of a Beta distribution whose evidence is the shares seen (each round counting 4 against the uniform guess of round 1, older rounds weighing 0.8 as much as the next). A move nobody tried keeps some weight on higher shares, so it can still be tried.

The guess replaces a self-referring question ("how many will follow, given that each follows if enough follow?") whose answer is not determined by one round's polls: nobody following is always a consistent answer, and so is a large following when the move is attractive. The uniform guess is the answer of a voter who cannot tell which of these will happen; the share actually seen then settles it.

For now voters see the true share of their side who cast the move; seeing it only through the noisy polls is a refinement for later.

### Memory: earlier rounds stay in mind

A move's value to a voter is a weighted average over the rounds in which it was weighed, each earlier round weighing 0.8 as much as the next, so a counter-move seen before keeps lowering the value of the move it answered. A move whose race is no longer among the poll's top three counts as gaining nothing in that round, so its value fades over a few rounds instead of vanishing the moment the polls reorder.

### The voter's decision

A voter is on three sides and has several moves in mind. They cast the move of greatest remembered value if that value exceeds their inertia (both as shares of their utility range); otherwise their honest ballot. Their inertia enters once, as the threshold for leaving home.

### What this replaces

- **No `--coordination`.** The share a voter imagines following starts unknown and is learned from what their side does.
- **One trait instead of several.** The cost of insincerity, the adaptive and always-strategic voters and the aggressive and kind attitudes all reduce to inertia.
- **No trust mechanism.** The current engine lowers its trust in moves that disappointed. Here every move is valued against the side at home on the current poll, which shows any backfire directly, and memory keeps earlier rounds in view.
- **No optimisation of participation.** Neither the side nor the voter works out who will take part.

### What Changed

The first version (2026-11-11) had sides choose a rule, a move together with a cutoff: the members whose inertia was below the cutoff cast the move, and the side chose the cutoff (of 10, 25, 50, 75 or 100% of its members by inertia) with the largest total gain minus the members' inertia. Inertia applied to every change, measured from the ballot cast last. Two objections led to the revision:

- **The cutoff is an optimisation no voter can do.** It requires knowing the inertia of everyone on one's side. Voters decide independently and must guess.
- **Inertia from the last ballot spirals.** With the guess in place of the cutoffs, burial in Condorcet methods grew from round to round (Schulze: 10, 16, 19% in rounds 1, 3, 6), since harmless burial never gains enough to be taken back. Measured from home it stays level.

## Not Covered

- **Strategy without any reason.** Voters who bury a rival out of habit, whatever the polls say, are outside the model. If wanted, they are a separate small layer; it sets directly how much each method is hurt, so it comes last and needs data.
- **How finely polls are read.** With 5 polls, winner chances come in steps of 0.2. Reading the polls as 40 synthetic polls with their mean and spread was tried in the first version and changed little (below); the averaging over shares of the guess smooths a move's value in a similar way.
- **Moves for a third candidate's own win** beyond the push-over, and campaigns, which could become the way a side's rule is announced.
- **Strategic nomination** still foresees sincere voting.

## Cost

Each move is counted at four shares, plus once for the side at home, for six sides: about 25–30 counts of the round's polls per round and method. In trials the engine takes about as long as the first version and about twice as long as pooled discovery. Counting dominates the run time; the memo of equal changes helps. The cost grows with the number of rounds; rank correlations in the first version showed the order of the methods settled after round 1, so 3 rounds are a reasonable standard.

## Checks Before Adoption

1. **Calibration.** Under Plurality at the defaults, the share of voters deserting a hopeless favourite should stay near what surveys find (4–8% of all voters; about one in eight of a third party's supporters). This sets the distribution of inertia, the one assumption of the engine.
2. **Full strategy.** With every voter at θ = 0, the results should be close to those of the current engine at `--coordination 1` (the table above), without any setting. Such voters have no inertia, so nothing damps them and they may cycle without settling; the report should show how often, rather than the engine preventing it.
3. **Approval.** Strategic approval voting should move the result towards the Condorcet winner, as theory expects; with the current engine it barely moves.
4. **Known answers.** Plurality with three candidates should show Duverger's desertion. A Condorcet method with a clear Condorcet winner should show burial among voters with low inertia in the early rounds, and some of it taken back once the buried side's answer shows in the polls.
5. **Damping.** The share of voters changing their ballot should fall over the rounds, and the share of elections whose ballots settle should not fall compared with the current engine.
6. **Robustness.** The results should not depend much on the constants of the guess and the memory (the four shares, the evidence of 4 per round, the weight 0.8 of earlier rounds), and their dependence on the distribution of inertia should be measured and reported.
7. **The method order** should be explainable where it differs from version 2.x.

## Measurements of the First Version (2026-11-11)

The first version (rules with cutoffs, What Changed): sides per race among the poll's top three; the naive exaggerations, the push-over and the honest ballot as moves; cutoffs at the lowest 10, 25, 50, 75 and 100% of a side's members by inertia; inertia uniform from 0 to `--inertia` for adaptive voters (0 for always strategic ones). It does no random search and has no campaigns. A new results column, `changed`, gives the share of all voters whose ballot differs from the round before (every engine). Trials at the defaults (100 elections, seed 1; pooled and scale 0.3 with the standard set and rounds 1 to 6, scales 1 and 3 with Plurality, IRV, Approval, Score, STAR, Schulze and Soft Minimax-D and rounds 1, 3 and 6):

| | pooled | inertia, scale 0.3 | inertia, scale 1 | inertia, scale 3 |
|---|---|---|---|---|
| voters changing their ballot per round (rounds 1–6) | 5–8% | 21–24% | 5–6% | 1–2% |
| ballots settled (rounds 1–6) | 41–57% | 17–21% | 28–34% | 50–59% |
| Plurality: voters deserting their favourite (rounds 1, 3, 6) | 6, 5, 2% | 18, 22, 23% | 7, 9, 10% | 2, 3, 3% |

- **Calibration.** At scale 1 the desertion under Plurality is near the survey range (4–8% of all voters), at 0.3 it is three times that. The default of `--inertia` is therefore 1: an adaptive voter's inertia is uniform between nothing and their whole utility range.
- **Damping.** At scale 0.3 a fifth of the voters change their ballot every round and burial grows over the rounds: the sides act at the same time, each valuing its rule as if everyone else voted as the poll shows, a whole side's burial or compromise moves the winner chances of five polls by steps of 0.2, and the promised gains are far above most inertias. At the calibrated scale 1 the engine reaches a steady level instead. Of all voters per round (40 elections, seven methods, rounds 1 to 6), honest voters taking up a move fall from 6.6% to 1.6%; voters going back to their honest ballot (about 1%) and voters switching from one move to another (about 2%) stay level, and by round 6 they nearly balance the newcomers. What keeps moving is the low-inertia voters answering each round's new polls. As a share of voters changing per round this is no more than with pooled discovery (5–8%, not falling either); fewer elections settle only because the changes are spread over more of them. "Settled" (no ballot changing at all) is a strict measure for an engine with fresh polls every round; the share changing and the flow between honest and strategic ballots say more.
- **Smooth reading of the polls** (tried, then removed from the code): reading the polls as 40 synthetic polls with their mean and spread, the same draws for every move of a round, so that a move's chances change with how far it shifts the counts. At scale 1 it made the share changing per round only a little lower (4.0 instead of 4.7%, rounds 1 to 6), with full strategy higher (63 instead of 50%), changed no result beyond the noise of 100 elections and took four times as long. The jumps of a five-poll reading are not what keeps the voters moving.
- **Full strategy** (inertia 0 for everyone, `--strategic 1 --adaptive 0`, 4 methods): Schulze 80.6 / 88.7, Soft Minimax 82.2 / 89.9, IRV 86.6 / 86.9, Plurality 84.1 / 81.1 (VSE\* / WME\*); about half the voters change their ballot every round. Against pooled discovery at `--coordination 1` (the table above) Schulze is close, Soft Minimax and IRV lower: without inertia the sides cycle, as expected.
- **Results at scale 1** (round 3, VSE\* / WME\*; pooled in brackets): Plurality 80.9 / 75.8 (80.4 / 75.4), IRV 91.4 / 93.3 (91.1 / 88.8), Approval 90.1 / 89.2 (90.4 / 87.9), Score 87.7 / 95.8 (84.8 / 90.3), STAR 90.1 / 93.7 (85.9 / 95.4), Schulze 88.6 / 98.2 (88.0 / 99.9), Soft Minimax-D 89.5 / 99.3 (89.5 / 99.3). With 100 elections differences of about 2 points are within the noise.
- **Cost:** about twice the time of pooled discovery.

For the first version, checks 1 (calibration), 2 (full strategy) and 5 (damping) were done in a first form and 6 (cutoffs, distribution) started; 3, 4 and 7 are open. Check 7 needs the standard set at the usual size beside pooled discovery.

## Measurements of the Revised Version (2026-11-12)

100 elections at the defaults (seed 1, the standard set, rounds 1 to 6, inertia scale 1). The steps from the first version, each with the one before:

| | voters changing per round (rounds 1, 3, 6) | Plurality: deserting their favourite | Schulze: burial |
|---|---|---|---|
| guess (unknown share), inertia from the last ballot | 11, 10, 10% | 6.5, 8.3, 9.3% | 10.0, 15.5, 18.8% |
| reasoning from home | 11, 13, 13% | 6.5, 6.6, 6.4% | 10.0, 10.6, 10.7% |
| + memory (earlier rounds weighing 0.5) | 11, 12, 11% | 6.5, 6.5, 6.2% | 10.0, 10.7, 10.7% |
| + learning the share | 11, 11, 10% | 6.5, 5.6, 5.1% | 10.0, 9.0, 7.8% |
| + memory 0.8, moves of races that drop out fading (the engine) | 11, 9, 6% | 6.5, 6.2, 5.3% | 10.0, 9.8, 8.2% |

- **Reasoning from home** stops the growth of strategy over the rounds.
- **The churn** left after that came from voters leaving home, returning and switching moves in about equal parts (about 3% each per round). Much of it followed the poll's top three reordering: the moves of a race that dropped out vanished, and their voters went home until it came back. Letting those moves fade from memory instead halves the churn by round 6, and it keeps falling.
- **Calibration:** at scale 1 the desertion under Plurality (5–7%) is within the survey range (4–8%), so `--inertia` stays at 1.
- **Burial** is common in the Condorcet methods and IRV (about 8–10% of the voters), three times as much as with the cutoffs, which held it back by charging each side for its members' inertia. It does little harm: Schulze's WME\* stays at 99.5. There are no data on burial to fit it to, so if a realistic model shows it common, that is a result.
- **Full strategy** (inertia 0 for everyone, `--strategic 1 --adaptive 0`, four methods, rounds 3 and 6): the share changing falls from 19.5 to 12.7% (first version: about half, not falling). Every voter with any gain makes a move, so burial reaches 59% of the voters in Schulze and IRV and compromise 47% in Plurality. VSE\* / WME\* in round 6: Schulze 89.2 / 98.2, Soft Minimax 86.7 / 96.3, IRV 86.8 / 91.4, Plurality 80.6 / 78.3 (pooled discovery at `--coordination 1`: 82.5 / 91.3, 86.2 / 97.6, 86.7 / 94.5, 78.2 / 79.5). The Condorcet methods hold up under full strategy here, unlike with the first version.
- **Results** (VSE\* / WME\*; honest; first version round 3; this version rounds 3 and 6): Plurality 75.1 / 72.1; 80.9 / 75.8; 75.8 / 72.8 and 71.3 / 69.1. IRV 90.8 / 94.3; 91.4 / 93.3; 90.5 / 92.8 and 90.8 / 94.3. Score 86.6 / 94.6; 87.7 / 95.8; 87.1 / 93.7 and 88.2 / 92.6. STAR 90.9 / 97.7; 90.1 / 93.7; 91.6 / 98.3 and 92.6 / 96.4. Borda 84.8 / 94.3; 89.5 / 92.4; 90.8 / 94.7 and 90.8 / 95.3. Schulze 88.4 / 100.0; 88.6 / 98.2; 86.5 / 99.5 and 87.6 / 99.5. Soft Minimax-D 89.5 / 99.3; 89.5 / 99.3; 89.1 / 99.1 and 89.1 / 99.1. With 100 elections differences of about 2 points are within the noise; the drop of Plurality below its honest value needs the larger run to confirm.

## Open Questions

- Sides per race, or types (above).
- The constants of the guess and the memory (Robustness), and whether voters should see the share their side cast only through the noisy polls.
- The distribution of inertia: its shape (uniform, or with many voters at high inertia), and whether it should depend on how far a move is from the honest ballot (a compromise for a near favourite costing less than burying someone).
- Whether the layer of strategy without any reason is needed at all.
- **Full strategy** (everyone at inertia 0) damps but stays restless: a fifth of the voters change in round 3, an eighth in round 6 (Measurements). Whether it should settle further, for instance by valuing a move against the other side's best answer in the same race (one step of looking ahead), depends on how much that case matters.
