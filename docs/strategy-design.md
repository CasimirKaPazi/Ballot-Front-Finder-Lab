# Strategy: Design Note for the Next Engine

Status: a proposal under discussion (2026-11-10, revised 2026-11-11). Nothing here is implemented. It would be a further engine (`--engine inertia`, a module of the `strategy/` package next to the others, in the poll rounds they share); the existing engines stay for comparison. The engine in use is described in [model.md](model.md) (Individual voters); this note says what is wrong with it, proposes one principle to replace its separate motivations, and lists what is still open.

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

> A voter casts the ballot they would want everyone on their side to cast, and changes their ballot only when the polls show that changing gains them more than their inertia; their honest ballot is where they start.

The first half is the rule-utilitarian (or ethical) voter of the literature on turnout (Feddersen and Sandroni 2006; Coate and Conlin 2004), applied to the choice of ballot; that application is this note's own step. The second half is inertia, or satisficing (Simon 1955): a voter keeps what they have until something clearly better is on offer. The poll rounds then do the work of reasoning about the others: each poll shows what the others did in answer to the last one, and voters respond to it as far as it moves them.

## The Model

The model has three parts: who is on a voter's side (sides), what a side does (its rule), and when a voter changes their ballot (inertia). Voters differ in **one trait only**, their inertia.

### Sides: one per race

The polls give the candidates likely to win (the top three, as now). Each pair of them is a race, and in each race a voter has a side: everyone who prefers the same candidate of the pair. A voter with C > B > A is on B's side against A, on C's side against B and on C's side against A. With three places there are three races and six sides, and every voter is on three of them.

A side is defined by one preference only, as real camps are: "everyone who wants to stop A" includes voters who disagree about everything else.

*Alternative:* the six types of the current engine (the voters with the same order of the top three), each with one rule. Simpler, but it splits a camp into groups that cannot see their joint strength. The per-race sides are the proposal; the types remain the fallback if the sides prove too costly or unstable.

### Inertia

Each voter has an inertia θ, a share of their own utility range, drawn per election from their own random stream (the same for every method). They see every poll. In each round they compare the best ballot available to them (below) with the ballot they cast in the last round, and change only if the gain in their expected outcome over the round's polls, as a share of their utility range, exceeds θ. Everyone starts from their honest ballot.

- **Low inertia:** reacts to small advantages, and takes a move back as soon as it disappoints.
- **High inertia:** stays honest unless the polls show things going badly enough that a change gains a lot. A poll much worse than the voter expected shows up as a large gain from changing, so no separate notion of what each voter expected is needed.
- **In between:** a voter may make a move when the stakes are high and keep it after a mild backfire, since taking it back gains less than their inertia. The naive strategist, who acts on a first impression and does not correct it, comes out of the dynamics rather than out of a separate trait; so do the aggressive and kind voters of version 1.0.0.

Inertia applies to **every change**, leaving the honest ballot and returning to it alike. The alternative, a cost paid only for departing from the honest ballot (returning free), is the current engine's cost of insincerity: it has no memory, so voters who stop gaining fall back to honesty at once, and honesty is a cost paid every round rather than a default. Inertia on every change makes honesty the default and gives the voters' history a place.

Inertia is smooth at the level of the electorate: each voter has a sharp threshold, but with θ drawn from a continuous distribution, the share who change grows smoothly with the gain on offer. A probabilistic change per voter would add a second source of randomness without a second idea.

### A side's rule: a move and a cutoff

For each side, the obvious moves for its candidate against the other are tried, as now (the naive exaggerations and the push-over, each voter casting the move from their own utilities); returning to the honest ballot is always among them, so a side whose earlier move backfired can take it back. New is who takes part. A rule is a move together with a cutoff: *everyone on this side whose inertia is below the cutoff casts the move*.

- For each move and each of a few cutoffs, the ballots of exactly those voters are changed and the round's polls counted again with the method. Everyone else is taken to vote as the poll shows.
- The rule's worth is the side's total gain (each member's expected utility over the polls, by their own utilities over all outcomes, as a share of their utility range) minus the inertia of the members who would have to change. Members whose last ballot already follows the rule change nothing and cost nothing, so a side that is already acting does not count its effort twice.
- The side's rule for a move is the cutoff with the largest worth; if no cutoff has a positive worth, the move is not offered.

Members of a side differ in what they think of the other candidates, so a move can help some and harm others; the total takes that into account. Nobody assumes anything about the other sides beyond what the poll shows: what they actually did is visible in the next poll.

### The voter's decision

A voter is on three sides and may be offered several rules. They consider the rules whose cutoff is above their own inertia, together with their honest ballot and the ballot they cast last. Each is valued by what it promises *them* over the round's polls (for a rule: if it is followed as planned). The voter casts the most valuable one if it gains more than their inertia over the ballot they cast last; otherwise they keep that ballot. Their inertia enters once, as the threshold for changing.

### What this replaces

- **No `--coordination`.** How many move follows from the voters' inertia and from what the move achieves at each level of participation.
- **One trait instead of several.** The cost of insincerity, the adaptive and always-strategic voters, trust in moves and the aggressive and kind attitudes all reduce to inertia. Always-strategic voters are those with θ = 0.
- **Assumed and actual participation agree** up to the voters who prefer another side's rule.
- **Moves that need many voters are found** if they are worth the inertia of those who must change (burial that pays only with most of a side), and moves that overshoot at full participation are offered at the level where they work.
- **No trust mechanism.** The current engine remembers what each move promised and lowers its trust in moves that disappointed. With inertia, every voter values every option against the current poll, which shows any backfire directly; whether they act on it depends on how large it is against their inertia.
- **Damping.** Small swings in the polls move nobody whose inertia is above them, so the back-and-forth of the current engine should settle where no voter gains more than their inertia from changing: a satisficing equilibrium.

An earlier draft of this note gave voters a second trait, attention: how many poll rounds they keep following (levels of reasoning, after Camerer, Ho and Chong 2004). It was dropped because a voter who never looks at the polls and one who looks but never finds a change worth it cast the same ballots, so the two traits could not be told apart in the results or calibrated separately.

## Not Covered

- **Strategy without any reason.** Voters who bury a rival out of habit, whatever the polls say, are outside the model. If wanted, they are a separate small layer; it sets directly how much each method is hurt, so it comes last and needs data.
- **How finely polls are read.** With 5 polls, winner chances come in steps of 0.2, so a move is worth nothing or a lot. That is a property of the valuation, separate from the noise of the polls: the 5 polls could stay as the information and moves be valued on a smoothed reading of them.
- **Moves for a third candidate's own win** beyond the push-over, and campaigns, which could become the way a side's rule is announced.
- **Strategic nomination** still foresees sincere voting.

## Cost

Each move is counted at several cutoffs (about 5), for six sides instead of six types, so about 5 times the counting of the current engine per round. Counting dominates the run time; the memo of equal changes and the fast subset totals help. Since the ballots should settle, more rounds than now (4 or 5) may be needed to reach the end state, but later rounds can skip sides none of whose members would change. A first implementation should measure this on the standard set before anything else.

## Checks Before Adoption

1. **Calibration.** Under Plurality at the defaults, the share of voters deserting a hopeless favourite should stay near what surveys find (4–8% of all voters; about one in eight of a third party's supporters). This sets the distribution of inertia, the one assumption of the engine.
2. **Full strategy.** With every voter at θ = 0, the results should be close to those of the current engine at `--coordination 1` (the table above), without any setting. Such voters have no inertia, so nothing damps them and they may cycle without settling; the report should show how often, rather than the engine preventing it.
3. **Approval.** Strategic approval voting should move the result towards the Condorcet winner, as theory expects; with the current engine it barely moves.
4. **Known answers.** Plurality with three candidates should show Duverger's desertion. A Condorcet method with a clear Condorcet winner should show burial among voters with low inertia in the early rounds, and some of it taken back once the buried side's answer shows in the polls.
5. **Damping.** The share of voters changing their ballot should fall over the rounds, and the share of elections whose ballots settle should not fall compared with the current engine.
6. **Robustness.** The results should not depend on the grid of cutoffs, and their dependence on the distribution of inertia should be measured and reported.
7. **The method order** should be explainable where it differs from version 2.x.

## Open Questions

- Sides per race, or types (above).
- The grid of cutoffs, and whether a side may offer more than one rule.
- Whether the side's total should weigh members equally (as proposed) or by how much they care.
- The distribution of inertia: its shape (uniform as the current cost, or with many voters at high inertia), and whether it should depend on how far a move is from the honest ballot (a compromise for a near favourite costing less than burying someone).
- Whether the layer of strategy without any reason is needed at all once inertia is in.
