# Measures and analysis

What is measured and how `analyze.py` reports it. How to run the simulations is in [running.md](running.md).

## Axes and other outputs

Both axes are **efficiencies with the same two reference points**, as ratios of sums over all elections:

`E = Σ(X_method − X_rb) / Σ(X_best − X_rb)`

- **0 = random ballot**: the winner is the favourite of a randomly drawn voter. It needs no assumption about which candidates stand, and every reasonable method should beat it.
- **100 = the best candidate** of each election. Values above 100 cannot be reached; the plots grey that area out. Values below 0 are worse than random ballot.
- The zero is a common level for all elections; it does not judge individual results. Compromise candidates are therefore still rewarded: in an election where the best candidate is nobody's favourite, electing it counts fully toward the numerator.

**VSE** (voter satisfaction efficiency), first axis: X = total utility of the winner, `U(c) = Σ_v u_v(c)`; best = the candidate with the highest total utility. The usual VSE puts the zero at a random candidate instead. With candidates drawn from the same distribution as voters the two differ only in scale, so the order of methods is the same. No utility normalisation: a voter who cares little (all candidates far away) counts little.

**WME** (worst-margin efficiency), second axis, a refined Condorcet measure: X = the winner's worst margin, its smallest head-to-head margin against any other candidate, by sincere preferences, as a share of voters. Best = the candidate with the best worst margin.

- The motivation is legitimacy: after the election voters judge the result, and if someone other than the Condorcet winner won, they can argue that a majority would have preferred another candidate. The worst margin measures how strong that argument is, instead of only whether it exists: losing a Condorcet winner by one vote costs little, electing a candidate whom a large majority prefers a rival to costs a lot.
- Every election counts, cycles included, and the value does not depend on the number of voters.
- With honest ballots Minimax reaches 100 by construction (it elects the candidate with the best worst margin), and all Condorcet methods reach it whenever there is a Condorcet winner, just as they reach 100 on CE. The strategic version WME\* is where methods are tested.

**Reachable frontier.** VSE and WME pull apart: the candidate with the highest total utility is often not the one with the best worst margin, so no rule can reach 100 on both. `analyze.py` computes what any rule could reach in the simulated elections if it knew every voter's sincere utilities: for each weight λ between 0 and 1 it elects, in every election, the candidate with the highest λ × (its VSE contribution) + (1 − λ) × (its WME contribution). Since both axes are ratios of sums with denominators that do not depend on the rule, this traces the exact boundary of the reachable region (lotteries fill in between the points). The plots draw it as a curve and grey out the region beyond it; the report gives its two ends and its most balanced point (highest VSE + WME).

**Reachable region.** The frontier is only the upper-right edge of what can be reached. The whole region is a convex polygon: every rule elects one candidate (or a lottery) per election, so its (VSE, WME) is a point inside the polygon spanned by all combinations. `analyze.py` computes it the same way, for all directions round the circle instead of the upper-right quarter only: for each direction it elects, in every election, the candidate furthest in that direction, and these points are the polygon's corners. The plots grey out everything outside it. This shows that the trade-off is pinned at its ends too: a rule with WME 100 always elects the candidate with the best worst margin, which gives one VSE (up to ties), not anything lower. So a point like VSE 50 / WME 100 is not merely bad but impossible, and the region narrows towards both ends. The report gives the other axis at each end ("with the best WME (100) VSE …"). Two reference points are marked with small crosses where they fall inside a plot: the random ballot (0 / 0 by definition) and a random candidate (the zero of classical VSE; its WME is the average worst margin of the candidates). The report gives the random candidate's values.

A method's distance to the curve is what it loses to having only ballots, not utilities, and to strategy. Faint dashed lines mark 1, 2, 4 and 8 points below the frontier (labelled −1 … −8 at their lower end): each is the frontier shifted down and left by that many points on both axes, so a method on the −4 line could gain 4 points of VSE *and* 4 of WME before it reached the frontier. A strategic shift parallel to these lines is a trade between the two measures; one that crosses them inward is a loss on both. They are drawn inside the region only: beyond the frontier's ends the region bends away. Without a frontier (chain and track plots over conditions with different electorates) the lines follow the 100 / 100 corner instead. For this, run.py stores every candidate's total utility and worst margin per election (columns U_cands, WM_cands).

**Condorcet efficiency (CE)**, reported alongside but not an axis: among elections with a Condorcet winner by sincere utilities, the probability the method elects it. It ignores elections without a Condorcet winner and does not distinguish narrow from clear ones.

VSE, WME and CE use sincere ballots; VSE\*, WME\* and CE\* (in the CSV: VSE_learned, WME_learned, CE_learned) the strategic ballots ("learned": after the poll rounds, or after convergent learning). In all cases the preferences that count are the sincere ones.

**Output files of `run.py`:** `elections.csv` has one row per (election, method, ballots = sincere or learned) with the run's parameters as columns; the simulator version (column `version`, from `run.VERSION`; the report names it, and warns when CSVs of different versions are combined); the quantities for the axes: U_method, U_best, U_rb (random ballot), U_rand (random candidate, for the usual VSE), WM (the winner's worst margin, share of voters), WM_best, WM_rb; has_cw and p_cw (probability of electing the Condorcet winner); the electorate properties cw_is_util, U_cw and fav_ratio (the same in every row of an election; see Robustness); sincerity, settled and the strategy types. `<out>-summary.csv` and the printed table have one row per method:

| Column | Meaning |
|---|---|
| VSE, WME | voter satisfaction and worst-margin efficiency, sincere ballots |
| VSE_learned, WME_learned | the same with learned ballots |
| CE, CE_learned | Condorcet efficiency, sincere and learned ballots |
| sincerity | mean sincerity of the learned ballots, over all voters (below; elections.csv also has `fidelity`, for both ballot sets) |
| settled | poll rounds: share of elections where the last round gave the same ballots as the round before (a fixed point); convergent learning: no group changed its ballot in the last round. Cycles never settle |

`on_front` in the summary is a quick check on point estimates (VSE_learned, WME_learned, sincerity; no uncertainty). The analysis uses the tie-aware "Dominated by" instead (see Analysis).

**Strategy types** (`strategy.strategy_types`), among voters whose learned ballot reverses at least one real preference:
- *compromise*: the favourite is not at the top, and the best-liked top candidate is liked above the voter's mean (a viable second choice raised);
- *push-over*: the favourite is not at the top, and every top candidate is liked below the mean (a weak candidate raised, to steer eliminations or runoffs);
- *burial*: the favourite stays at the top, but a preference further down is reversed.

The three are exclusive. For score ballots, "burial" includes small reversals among lower-rated candidates, so a high burial share can go with high sincerity.

**Sincerity and fidelity.** Two separate questions about a learned ballot: does it go against the voter's own preferences, and how much does it tell about them?

**Sincerity** (`strategy.sincerity`) is about voting against one's own preferences because it pays. Per voter, the *reversal mass* R of a ballot is the sum of `u[a] − u[b]` over the pairs the voter prefers a > b but the ballot puts b above a. Sincerity is `1 − R(ballot) / R(worst)`, where the worst ballot is the most dishonest one the method's ballot format allows for this voter: a vote for the last choice in plurality, a veto against the favourite, the reversed ranking (rankings, and scores with enough levels), the most damaging cut in approval (found exactly, over all cuts), the reversed order in three levels for three grades, the three least liked ranked in reverse for IRV-3. So 1 means no reversed preference and 0 means as far against one's own preferences as this format permits, the same in every format. Ties are not reversals and cost nothing: exaggeration (5, 3, 1, 0 → 5, 5, 0, 0), an approval threshold anywhere, unranked candidates. They say less, not something false, and in approval or plurality there is no single honest place for the threshold anyway. This is the usual definition of a sincere ballot (Brams and Fishburn: no candidate is put above one the voter likes more), scaled per format.

**Fidelity** (`strategy.fidelity`) is about how much of the voter's utility differences the ballot conveys. Utilities are stretched to 0–1 per voter (favourite 1, least liked 0) and compared with the values the ballot implies: scores and grades by their value, rankings evenly spaced, approval and plurality marks as 1 and 0, veto 0 for the vetoed candidate. Over all pairs, fidelity is `1 − Σ |Δu − Δv| / Σ 2 |Δu|` (Δu, Δv the differences in utility and implied value): 1 if the ballot shows the differences exactly; a tie loses half of a pair; a mirrored pair loses all of it. A sincere ranking of utilities 100 / 3 / 2 / 1 / 0 loses a lot, since it implies 100 / 75 / 50 / 25 / 0. Fidelity therefore includes what the format cannot express: the report gives it for the learned ballots and for the sincere ones ("(honest)"), so the format's own loss and what strategy adds can be told apart. It is reported for context, not used for the front.

Example, utilities 0.5 / 0.3 / 0.15 / 0.05:

| Ballot | Sincerity | Fidelity |
|---|---|---|
| Plurality: vote for the 1st choice (sincere) | 1 | 0.72 |
| Plurality: vote for the 2nd choice | 0.75 | 0.32 |
| Plurality: vote for the last choice | 0 | 0.05 |
| Ranking, sincere | 1 | 0.93 |
| Ranking: 2nd moved above the 1st | 0.87 | 0.63 |
| Ranking: reversed | 0 | 0 |
| Score 0–5: sincere 5, 3, 1, 0 | 1 | 0.97 |
| Score 0–5: exaggerated 5, 5, 0, 0 | 1 | 0.70 |
| Approval: {1st, 2nd} (sincere) | 1 | 0.70 |
| Approval: {3rd, 4th} | 0 | −0.10 |

The exaggerated score ballot and the approval ballot say the same and get the same values. Two-round ballots (TTR, the approval runoffs): the voter only chooses the round-1 marks (the runoff ranking is the sincere one), so both measures look at the marks, as a plurality or approval ballot. **Undefined for TRASH and the Inverse-\* methods.** Both measures assume that the honest ballot is the one a voter would want to keep, so moving away from it is a cost of strategy. In TRASH (the candidate ranked lower wins) and the Inverse-\* methods (honest voting elects the worst candidate) the honest ballot works against the voter's own interest: what a ballot says and whom it helps point opposite ways, and leaving the honest ballot is escaping a bad method, not giving up honesty. Sincerity and fidelity are therefore left blank for these methods (`methods.sincerity_defined`), in the CSV, the tables (—) and the plots; on the front they count as ties on the sincerity axis. The strategy types (compromise, burial, push-over) are still reported; for the Inverse-\* methods they are measured against the reversed preferences the honest ballot expresses. CSVs from before 2026-10-12 (no fidelity column) hold an earlier sincerity measure (reversals over all pairs, not scaled per format); the report says so.

**Condorcet winner in the cast ballots** (`metrics.ballot_condorcet`). Studies of real ranked-ballot elections compute the Condorcet winner from the cast ballots, which may themselves be strategic. The model knows both: the real Condorcet winner (sincere preferences) and the one the ballots show (a voter counts for a over b if the ballot puts a higher; equal = no preference). Per election and method (columns `bcw`, `p_bcw`): the ballots show the same Condorcet winner, another one (false), none although there is a real one (a cycle or ties), or one although there is none (spurious); and how often the method elects the ballots' Condorcet winner, which is the Condorcet efficiency ballot data would report. Only for ballots that record an order (rankings, scores, grades); plurality, approval, veto and two-round marks are left blank, since their pairwise counts mostly reflect the marks. For the Inverse-\* methods the ballots are read reversed. The report has a table per condition, with the honest ballots in brackets: for rankings these always show the real winner, for scores and grades rounding and ties can already hide or change it.

## Analysis

`analyze.py` reads one or more elections CSVs. Rows with the same parameters except the seed form one **condition**; seeds are pooled as replications.

**Uncertainty: paired bootstrap.** All methods face the same electorates, so each bootstrap replicate redraws whole elections with replacement and recomputes every method on that same draw (1000 replicates by default). This keeps comparisons paired: two methods can differ clearly even when the interval of each value on its own is wide, because elections differ much more from each other than methods differ within an election.

**The front: VSE\*, WME\* and sincerity.** What counts is the result with strategic voters and how honest they can afford to be. Honest VSE and WME depend more on modelling assumptions (who votes how sincerely is itself a model) and are reported for context, not for dominance. Above each condition's table the report gives two sets. The **Pareto front**: methods no other method matches or beats on all three point estimates (at least as good on all, better on one), however small the difference. **Undominated**: methods no other method clearly beats (see Dominated by, below), so noise-sized differences count as ties. The two need not agree: a tiny advantage keeps a method on the Pareto front but does not protect it from a method clearly better on another axis. summary.csv marks the Pareto front (`pareto_front`). The table is sorted by VSE\*.

**What is reported per condition and method:**

| Output | Meaning |
|---|---|
| VSE, WME, VSE\*, WME\*, CE, CE\*, sincerity | point estimate and 95% interval of each value on its own |
| dominated by | methods clearly better on at least one of the three front axes (VSE\*, WME\*, sincerity) and clearly worse on none, each with the axes on which it is better; the report shows at most three, the CSV all. Clearly: the paired 95% interval of the difference excludes 0, smaller differences count as ties. Without this, methods level on one axis would be kept apart by noise-sized differences. Sincerity is measured very precisely, so at equal VSE\* and WME\* the more honest method dominates |
| adaptive voters (with `--adaptive`) | share of the adaptive voters who turned strategic, per method |
| paired differences (`--compare "A:B"`) | difference A − B per axis with a 95% interval; marked as a tie when the interval includes 0 |

**How to interpret:**
- The values are the result. Whether a method lies on the Pareto front is already interpretation: it depends on which axes count and on which methods are included. "Dominated by" is only a pointer to where to look.
- The axes are kept separate on purpose. Each carries its own value judgement (utilitarian or majoritarian, honest or strategic); no weighting into a single score.
- Read results **per condition first**. The strategic axes depend on how voters strategise: the number of poll rounds, the aggressive and strategic shares (and, with `--converge`, the group size; see [model.md](model.md), Coordination).
- An ordering between two methods only counts if their paired difference is not a tie.

**Plots** (in the analysis folder):

| File | Shows |
|---|---|
| `arrows-<c>.png` | VSE/WME plane (everything outside the reachable region greyed out), every method's shift from honest (open) to learned ballots (filled, labelled) in one picture; filled points coloured by sincerity (blue = sincere, through purple to dark red = less honest; the scale is capped at the 10th percentile of the plotted methods); faint dashed lines 1, 2, 4 and 8 points inside the frontier |
| `shift-<c>.png` | the same shifts as one row per method, VSE and WME panels; easier to read per method |
| `sincerity-<c>.png` | VSE\* and WME\* against sincerity |
| `fidelity-<c>.png` | every method's ballots from honest (open, at sincerity 100 and the fidelity its format allows) to learned (filled) in the fidelity / sincerity plane: going left is information lost to strategy (exaggeration, levelling), going down is preferences reversed; only for CSVs with fidelity |
| `tradeoff-<c>.png` | ternary map of the Pareto front: each point of the triangle is a weighting of VSE\*, WME\* and sincerity (corners: all weight on one axis), coloured by the Pareto method with the highest weighted sum; each axis stretched to the range the Pareto methods span on it (shown at the corners), so it shows the trade-offs within the front, not their absolute size. Pareto methods best for no weighting (in a dent of the front) are listed below. Methods hidden with `--hide` (or not in `--show`) are left out before the front is found |
| `axes-heatmap.png` | VSE\*, WME\*, sincerity, VSE and WME, methods × conditions (when there are several; methods present in every condition, sorted by VSE\*) |
| `track-<method>.png` | with `--track`: one method across all conditions in the VSE\*/WME\* plane (learned ballots filled, honest open). One varied parameter: a chain in its order; two (a grid, e.g. `--aggressive` × `--rounds`): lines along the parameter with more values, colour for the other; more: points with condition tags |
| `chain-<parameter>.png` | sweeps: for each parameter varied around the base (the first condition), each method's learned values as a chain of arrows through the parameter's values, light to dark; one panel per method group, all on the same scale. If the parameter only affects voters' strategy (strategic and aggressive shares, poll rounds, polls, group size), the electorates are the same and each chain starts at the method's honest point; otherwise the honest values form their own dashed chain |

Sweep plots appear automatically for every parameter that has at least two conditions differing from the base only in that parameter: a one-at-a-time sweep (`sweep.py --one-at-a-time`) or a grid over a single parameter. They show point estimates only; with few elections the zigzags in a chain are mostly noise. One far-off method (often Plurality) sets the shared scale for all panels; `--hide` or `--show` zooms in.

Methods are identified by labels, not colour. Random pair runoff, Anti-plurality and TRASH are left out of the scatter plots by default (`--hide`), since their distance compresses the scale; they stay in tables and heatmaps. `--hide` with no names shows all.

Sample size: with 160 elections per condition the intervals of single values are about ±3 points and many paired differences are still ties. For conclusions, several hundred elections per condition and at least two seeds are advisable.

## Robustness: one parameter at a time

To see how much a model assumption matters, vary each parameter on its own around a base condition and check whether the methods keep their order:

```
python sweep.py --one-at-a-time --noise-strength 5 3 7 --dims 6 10 --width 0.7 1.0 \
    --rounds 1 3 --aggressive 0.5 0 1 --strategic 0.5 1 --seeds 1 2 --elections 400 --out model
python analyze.py model                                   # -> output/model/analysis/
```

Conditions are labelled by how they differ from the base (`c1: base`, `noise strength 3.0`, ...).

`analyze.py` adds a **rank stability** table when there are several conditions: for each axis, the rank correlation (Spearman) between the methods' values in a condition and in the base (the first condition). The methods hidden in the plots are left out. Its first row is a noise reference: the agreement expected from sampling noise alone between two runs of the base condition (the base's even and odd elections correlated, then scaled up to full size with the Spearman-Brown formula `2r / (1 + r)`). A condition near or above that row orders methods as the base does; a value clearly below it means the parameter changes which methods do well. With few elections everything is noise, and VSE\* in particular needs many elections (several hundred per condition): strategic differences between methods are small. Many methods lie close together on each axis, so rank correlation is a strict test; the heatmap shows where the changes come from.

An **electorates** table (and a line in each condition's section) describes the generated electorates themselves, independent of the methods: how often a Condorcet winner exists, how often it also has the highest total utility, how much VSE electing it loses (ratio of sums, as VSE), and how steep preferences are (u(favourite) ÷ u(least liked), median over voters, then median over elections). The third column explains most VSE differences between Condorcet and rated methods; the last one is a concrete handle for choosing `--width`.
