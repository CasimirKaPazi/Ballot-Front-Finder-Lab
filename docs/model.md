# The model

How electorates, ballots and strategic voters are modelled. Running the simulator is in [running.md](running.md), the measures and the analysis in [analysis.md](analysis.md), the methods in [methods.md](methods.md).

## Electorates

Each election draws a fresh electorate from the model chosen with `--model` (`models.py`, one entry per model in `MODELS`). A model draws only the positions of voters and candidates (and, if it has them, how much each person weighs each dimension); distances, utilities and the two candidate parameters below are the same for every model. The model and its own parameters are written into every row and named in the report; runs from before models existed are read as `clusters`. Common to all models: the distance is Euclidean (each dimension scaled by the person's weight for it, where the model has weights), and the utility is `u = exp(-distance / width)`, absolute and not normalised: 1 at zero distance, towards 0 far away. A smaller `--width` makes voters care only about nearby candidates.

Five models are available: `noise` (the default), `clusters`, `field`, `ks` and `ks-quinn`. Each is described below, the default first.

**Defaults, and why** (2026-10-16; [journal.md](journal.md), Electorate models). The defaults were chosen so that the electorates are plausible against two kinds of real data and still separate the methods:

- `noise`, because it has the fewest assumptions of the models and gives a balanced electorate: `clusters` is clumpy (with few clusters, islands), `ks` is uniform and low-dimensional (two or three dimensions drawn, in which VSE and WME hardly pull apart), and `field` gives a featureless blob.
- Strength 5, 6 dimensions, width 0.7. A Condorcet winner exists in about 97% of elections, as in real ranked elections with 50 to 100 or more voters (96–98% in Myers' CIVS data). The candidates' map from approval ballots (`--sav`, below) is close to Alós-Ferrer & Granić's real one with 6 candidates, if somewhat more concentrated on the first dimension. Electing the Condorcet winner loses about 15% of VSE, so VSE and WME pull apart and the methods differ. From strength 5 to 7 the method order stays within its sampling noise, so the exact value matters little. Below 5 (with 10 dimensions and width 1.0, below 6) the electorates become easy: nearly every method is close to the best possible result.
- The models' own defaults are those of their best variant in the comparisons: `clusters` with 20 clusters, no Zipf, fixed total variance and salience per cluster with exponent 0.5; `field` and `ks` unchanged.
- Before 2026-10-16 the defaults were `clusters`, 10 dimensions with Zipf, 5 clusters, no salience, unscaled positions and width 1.0; to reproduce older results, give these explicitly (each results file records every parameter).

**Fractal noise** (`--model noise`, the default). Voters are dense where a fractal noise is high, like Perlin noise with several octaves, so they gather in clumps within clumps at every scale. There is no number of clusters, no decay law for the dimensions and no salience to choose:

- The noise, new for every election: five octaves of eight plane waves each, in random directions in the `--dims`-dimensional issue space. From one octave to the next the frequency doubles and the amplitude falls by 2^−0.5. (This is fractal noise built from random waves; Perlin's own lattice construction is impractical in more than three dimensions.)
- People are drawn from a standard Gaussian weighted by exp(`--noise-strength` × noise). At 0 that is a plain Gaussian blob; larger values give sharper clumps, and electing the Condorcet winner becomes costlier while a Condorcet winner stays about as frequent.
- Sampling: 2000 points start from the Gaussian; the strength is raised in ten steps, and each time the points are resampled by their gain in weight and then take five small random steps (Metropolis), so that clumps get their real width. Voters and candidates are distinct points drawn from the result.
- Positions are scaled so the dimensions' total variance is π²/6, as `--cluster-fixed-variance` does for clusters. Candidates are drawn like voters. Drawing one electorate takes about 0.1 s, longer than the other models, but little next to the methods' strategy.
- The octaves, waves, amplitude decay and sampling settings are fixed in `models.py` (`NOISE_*`). `--noise-strength` is the knob that matters.

**Voter clusters** (`--model clusters`):

- cluster centres: `--cluster-count` points, normal in `--dims` dimensions;
- cluster sizes: random weights from a flat Dirichlet distribution, so some elections have one dominant cluster and others are balanced;
- voters: each assigned to a cluster by these weights, placed normally around its centre with standard deviation `--cluster-spread` (per dimension, before the Zipf scaling);
- candidates: drawn from the same distribution as voters (cluster by the same weights, same spread), unless moved by the two candidate parameters below.

**Zipf dimensions** (`--cluster-zipf 1`; the default until 2026-10-16), after Chris Smith's Mixture of Zipf Gaussians ([article](https://cdsmithus.medium.com/simulating-elections-with-spatial-voter-models-1ff50892390), [code](https://github.com/cdsmith/spatial-voting), `src/Model.hs`): dimension i (1, 2, 3, ...) has standard deviation 1/i — for cluster centres, voters and candidates alike — so one issue dominates and the others matter less and less, without being absent.

- Scaling follows Smith's code: the standard deviation is 1/i, so the variance is 1/i². (The article's wording "proportional to 1/n" could also be read as variance 1/n; the code is unambiguous.) The total variance then converges (to π²/6), so beyond about 4 dimensions their number hardly matters.
- Each cluster applies the 1/i scaling to its own slightly shuffled order of dimensions (each rank perturbed by normal noise with standard deviation 0.1 × dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues they vary on most. Cluster centres keep the global order.
- Taken over from Smith: the 1/i scaling and the per-cluster shuffle. Not taken over: his other settings (10 clusters with uniform random weights, cluster centres uniform in ±3/i, a random spread factor per cluster, utility = −distance); our Dirichlet weights, normal centres, one spread and `exp(-distance / width)` stay.
- `--cluster-zipf 0` (the default) gives all dimensions standard deviation 1. Distances are longer there, so without `--cluster-fixed-variance 1` the same width gives steeper preferences.

**Where people stand, and what they care about** (two options of `clusters`):

- `--cluster-fixed-variance 1` scales all positions by one factor so that the dimensions' total variance is π²/6, the limit of Zipf dimensions, whatever their number. Without it and without Zipf, every added dimension makes all distances longer, so preferences get steeper at the same width; with it, more dimensions spread the same disagreement over more issues. (With Zipf the factor is close to 1.)
- Salience per cluster (`--cluster-salience-exponent s`, default 0.5) separates what people care about from where they stand: each cluster weighs the issues 1/i^s (s = 1: 1, 1/2, 1/3, …, Zipf, as Smith's dimensions; s = 0: every issue alike, no salience) in its own random order, in its members' and its candidates' distances, scaled to a mean squared weight of 1. Society can then disagree on many issues while each group mainly cares about a few, and groups differ in which. The positions are the same with and without it (its random draws come last).
- Meant together with `--cluster-zipf 0`: positions spread alike on every issue, salience differs by group. This is the default of `clusters`: `--cluster-count 20 --cluster-zipf 0 --cluster-fixed-variance 1 --cluster-salience-exponent 0.5`. (Until 2026-10-17 a separate `--cluster-salience none|cluster` switched it; exponent 0 gives exactly the electorates of `none`.)

**Scale-free fields** (`--model field`, experimental). Instead of imposing clusters or decay laws, opinions arise as fields over a social space, following what is known about how they spread:

- People live on a two-dimensional social space (who meets whom; a 64 × 64 torus), placed by a Lévy flight, so they are clustered at all scales.
- Each issue is a Gaussian random field over that space with power spectrum k⁻²: correlations that fall off with the logarithm of distance, as Borghesi & Bouchaud (2010, "Spatial correlations in vote statistics: a diffusive field model for decision-making") found in real vote statistics, i.e. structure at all scales without a characteristic one.
- A person's opinion on an issue is the field where they live (40% of the variance; Borghesi & Bouchaud find a third to a half shared) plus an own part, passed through tanh(`--field-kappa` × ·): a saturating response, as a spin in the Ising model. Small values give smooth, nearly Gaussian opinions; large values push every issue towards two camps.
- `--dims` issues with a fixed total variance (as `--cluster-fixed-variance` for clusters). `--field-salience g` makes what people care about a field too (weights exp(g × another field)), so neighbours share their priorities partly.
- The exponent, the shared share and the Lévy exponent are fixed in `models.py` (`FIELD_EXPONENT`, `FIELD_SHARED`, `FIELD_LEVY`): in first tests they hardly changed the electorates' properties. `--field-kappa` is the knob that matters: from the smooth end (like ks) to the polarised end (like clusters).
- Issues are independent fields, so no ideology (opinions on different issues going together) is built in; how to let it emerge (overlapping topics, Baumann et al. 2021) is open.

**Quinn's hierarchical model** (`--model ks`), from Jameson Quinn's VSE simulations ([vse-sim](https://github.com/electionscience/vse-sim), `src/vse_sim/voter_models.py`, `KSModel`, "kitchen sink"), with the parameters of its published results. In the clusters model a voter belongs to one cluster on all dimensions at once, so clusters behave like consistent ideologies or parties. Here the issues come in bundles, and factions form in each bundle separately, so cleavages cut across each other: a voter can be in the big faction on one bundle and a small one on another, and groups differ in how much they care.

- Issue bundles: the first weighs 1, each further one a random fraction (Beta(1, 3)) of the one before, until the weight falls below `--ks-cut` (0.2); within a bundle, each further dimension weighs a random fraction (Beta(1.5, 3)) of the one before, down to the same cut. So the number of bundles and dimensions varies by election (with the defaults mostly 2–4 dimensions in all), and few issues matter much.
- Factions, in each bundle independently: a Chinese restaurant process over voters and candidates. Each person starts a new faction with probability α / (i + α), i the number of persons before, else joins the faction of a random earlier person (α = `--ks-alpha`, 1.5). A few large factions and a tail of small ones; their number is not fixed.
- Each faction, per bundle: a position (normal, with a random variance c ~ Beta(3, 1.5)) and a caring level (Beta(3, 1.5)). Its members sit around its position, the tighter the more it cares (offset N(0, 1) × √(1 − caring) per dimension), and weigh that bundle by their caring.
- Distance: each person weighs each dimension by its importance × their faction's caring on that bundle, normalised per person (as in vse-sim). Utilities use `exp(-distance / width)` like every model here (vse-sim uses −distance), so the models differ only in the electorate's structure. Candidates are drawn exactly like voters; the two candidate parameters below apply as for any model.
- Two versions. `ks` is the lab's version, which may evolve: a change comes as a new parameter whose default keeps the earlier behaviour (so older runs stay comparable), or under a new model name. `--model ks-quinn` is kept as close to vse-sim's code as the lab allows and is not to be changed: it uses vse-sim's one random offset per person and bundle (see the last point) and has no `--ks-shared`. Both read `--ks-alpha` and `--ks-cut`. What still differs from vse-sim in both is common to every model here: utilities `exp(-distance / width)` instead of −distance, and the candidate parameters.
- `--ks-shared 1` (a test variant of `ks`, not in vse-sim): the factions of the first bundle are used in every bundle, so cleavages no longer cut across each other; positions and caring are still drawn per bundle.
- Written anew from the algorithm; `ks` has one change: vse-sim's code gives a person one random offset for all dimensions of a bundle (it reads the offset by bundle number), which looks unintended; here each dimension has its own. With the exp utilities and this change, results are not directly comparable with Quinn's published numbers. `--dims`, `--cluster-count`, `--cluster-spread` and `--cluster-zipf` do not apply to this model.

**Looking at the electorates.** `show_electorates.py` draws the electorates of one or more models without running any method, in a few seconds, to check by eye that they look as intended and to compare models:

```
python show_electorates.py --model clusters ks ks-quinn --out look       # -> output/look/
python show_electorates.py --model ks --ks-cut 0.05 --width 0.5 --out look-ks
```

It takes run.py's electorate parameters (one value each; `--model` may list several) and the same random streams, so election e of a seed is the electorate run.py uses. It writes `examples.png` (a few elections per model: voters coloured by their favourite and the lettered candidates, on the voters' first two principal components; the Condorcet winner ringed), `summary.png` (over 300 elections: variance explained by the first principal components and how many reach 70%, favourite ÷ least liked, the utilities, the distances, and the VSE lost by electing the Condorcet winner where it is not the utility winner) and `summary.txt` (the same as numbers). For the ks models, whose people weigh the dimensions differently, positions are scaled by the voters' average weights first: the space roughly as an average voter sees it.

With `--sav` it also maps the candidates the way Alós-Ferrer & Granić (2015, "Political space representations with approval data", *Electoral Studies* 39) mapped real parties and candidates from approval ballots in German field experiments, after Laslier's spatial approval voting (2006). This gives a dimensionality comparable with theirs (four or more dimensions to explain 70% for real voters):

```
python show_electorates.py --model clusters ks --cands 9 --sav --out look-sav
```

- Each electorate is drawn with `--sav-voters` voters (1000), and every voter approves honestly, as in the methods (Honest approval below: a personal threshold between their mean and their favourite; until 2026-10-17 everyone above their mean). `--sav-noise s` (default 1) first blurs each utility by a normal error of s × the voter's own spread of utilities, so approvals are partly chance, as in real ballots (candidates a voter hardly knows, mistakes). Without it the model's approvals are so consistent that, in most elections, some pair of candidates is approved together by nobody, which the map can only patch (below); at 1 no such pair remains and the map comes closer to the real ones (2026-10-17; [journal.md](journal.md), Laziness and approval against real ballots).
- For each pair of candidates, the overlap of the people approving them, relative to chance: r = N · a(c, d) / (a(c) · a(d)). Squared distances are proportional to ln(r_max / r) (the paper's formula with its smallest admissible α; the most overlapping pair is at 0). Classical multidimensional scaling then gives the inertia each dimension explains; only positive eigenvalues count.
- A pair nobody approves together would be infinitely far apart; it is given half a voter instead, and the share of elections where this happens is reported. Without noise this happens in most elections, which makes the map less reliable.
- `sav.png` and `summary.txt` show the cumulative inertia by number of dimensions next to the paper's four data sets (Messel: 9 parties, 8 candidates; Konstanz: 11 parties, 6 candidates). Use a similar number of candidates (`--cands`), since n candidates allow at most n − 1 dimensions.

**Candidates unlike the electorate.** Two parameters move the candidates relative to the voters; both act after all random draws, so the voters (and the candidates' own draws) are the same for every value and a sweep compares the same electorates with different fields.

- `--cand-dispersion s` (default 1): each candidate's distance from the voters' mean position is multiplied by s. Below 1 the candidates cluster toward the centre of the electorate; above 1 they are more extreme than the voters. This is the relative candidate dispersion of Merrill (1984), "A comparison of efficiency of multicandidate electoral systems", *American Journal of Political Science* 28, whose Condorcet efficiency of plurality and IRV falls as candidates cluster (centre squeeze: the central candidate is squeezed out of the first choices).
- `--cand-shift d` (default 0): all candidates move along the first, most important dimension by d standard deviations of the voters on it: a systematic bias, e.g. candidates more often from one social group.
- Both change how far the best candidate is from the random ballot, the zero of VSE and WME. With a strong shift, one candidate is nearly everyone's favourite, the random ballot is almost as good as the best candidate, and the efficiencies' denominators become small; compare methods within a condition, and use CE or raw rates across conditions.
## Ballots and sincere voting

| Ballot | Form | Sincere ballot |
|---|---|---|
| one name | one candidate | the favourite |
| ranking | ranking; may be truncated (the top k ranked, the rest unranked, tied last) | by utility, complete (truncated with `--lazy`) |
| approval | approve or not | the candidates above the voter's personal approval threshold, between their mean utility and their favourite (see Honest approval below; at least the favourite; with `--lazy` only those the voter cares about) |
| score 0–max | integer score per candidate (`--max-score`, default 5) | stretched to the voter's own utility range, rounded (with `--lazy` the others blank = 0) |
| two-round, plurality first round | one candidate marked for round 1, plus a ranking for the separate runoff | the favourite marked; ranking by utility |
| two-round, approval first round | approvals for round 1, plus a ranking for the separate runoff | approvals as for the approval ballot; ranking by utility |
| veto | one candidate against | the least liked |
| ranking, ties allowed | levels, equal = tied; level 0 shared by two or more = unranked (truncation) | strict ranking by utility (ties only arise from learning; truncated with `--lazy`) |
| truncated ranking | the top k ranked, the rest unranked (tied last) | rank the candidates the voter would approve (as for the approval ballot; with `--lazy`: only those the voter also cares about) |
| ranking of at most 3 | up to 3 ranked, the rest unranked (tied last) | the 3 best-liked (all, with 3 or fewer candidates; with `--lazy`: only those the voter also cares about) |
| full ranking (Inverse-\* methods) | full strict ranking, never truncated | by the reversed preferences |
| three grades | good / OK / bad per candidate | stretched to the voter's own utility range, rounded to 0–2 (as the score ballot, with three levels; not affected by `--max-score`) |

**Honest approval.** Each voter has a personal approval threshold between their mean utility and their favourite: on utilities stretched to 0–1, mean + d × (1 − mean), with d drawn per voter and election uniformly on 0–1 (own random stream, the same for every method; a strategic group's reference ballot uses its members' mean d). So a candidate's chance of being approved rises linearly from 0 at the voter's mean to 1 at the favourite: a gradient, not one cut for everyone, and an approval says how much the voter likes a candidate on average (in expectation the approval ballot is a score ballot above the voter's mean, randomly rounded). It applies to the approval ballot, the round-1 approvals of the approval runoffs and the truncated ranking (MDDA and MAMPO, where ranked = approved). Why: real voters approve 1.5–3 candidates almost regardless of the field (approval elections and experiments with 4–16 candidates), while the earlier rule, every candidate above the voter's mean, approved about 0.4 per candidate (6.4 of 16); strategy removed only a tenth of that. Of the rules compared on the whole curve of sorted approval shares in 11 real elections, this one fitted best, without a fitted constant, a little better than vse-sim's BulletyApproval60 (60% bullet votes) ([journal.md](journal.md), Laziness and approval against real ballots). Until 2026-10-17 the threshold was the voter's mean; it changed the honest results of the approval methods considerably (Approval's honest VSE about 78 → 87).

**Truncation and laziness.** Rankings may be truncated: the voter ranks the top k candidates and leaves the rest unranked, tied last (a ranking of all but one is the full ranking). Strategic voters can truncate (rank only down to the leader they prefer, or bullet-vote for it) and the random search can rank one candidate more or fewer. Truncation reverses no preference, so it does not lower sincerity; it lowers fidelity (less is said). How the methods treat unranked candidates:

| Methods | Unranked candidates |
|---|---|
| pairwise (Schulze, Ranked pairs, Minimax family, Kemeny-Young, Copeland, Copeland//Margins, Baldwin, Nanson, SICT, Condorcet-random-drop, …) | below every ranked candidate, tied with each other (no preference between them) |
| IRV, ER-IRV, Benham, BTR-IRV, Coombs (first choices) | a ballot with no ranked candidate left is exhausted and counts for nobody; majorities are of the continuing ballots |
| Coombs (last places) | the unranked share the last place equally |
| Borda, Black, BTR-Borda | 0 points; ranked candidates keep their full points (C−1, C−2, …), so truncating can pay |
| Dowdall | 0 points |
| Supplementary vote | a ballot with only one candidate ranked has no second preference |
| TRASH (last places) | shared, as in Coombs |

In the ER variants (rankings with ties) a tie at level 0 of two or more candidates means unranked; ties higher up keep the equal-ranking rules (in ER-IRV each candidate tied at the top among the remaining gets a whole vote). The Inverse-\* methods keep full strict rankings: their honest ballot is the reversed one, and truncating it has no clear meaning.

**Laziness** (`--lazy λ`, 0–1) makes honest ballots truncated:

- **Each voter's own cut.** Each voter has a cut c, drawn per election uniformly between 0 and λ (own random stream, the same for every method). They rank a candidate only if its utility, stretched to 0–1 between their least and most liked candidate, is at least c (the favourite always). c = 0 gives a full ranking, c = 1 a bullet vote; in between the length depends on the voter's preferences (a voter with one clear favourite ranks fewer than one who likes several about equally).
- **What λ means.** λ = 0 gives full ballots, λ = 0.5 voters from complete to cutting at the middle of their range, λ = 1 the whole spread down to bullet votes. Voters differ, as on real ballots; a single cut for everyone made every format say roughly "these candidates are acceptable", and the methods alike (until 2026-10-16; [journal.md](journal.md), Laziness per voter).
- **Per ballot type.** It applies to every ballot that can say more than one thing. Rankings (ranking, ranking with ties, truncated ranking, ranking of at most 3; for the last two on top of their own cut) are truncated. Approval ballots (also the round-1 approvals of the approval runoffs) approve only the candidates above their threshold that the voter also cares about, so a cut of 1 is a bullet vote. Score and grade ballots leave the other candidates blank, which counts as 0, so a cut of 1 scores only the favourite. Plurality, veto and the TTR mark already name a single candidate. The Inverse-\* methods are never lazy.
- **Which value.** The default is 0 (full ballots). Real ranked ballots ([journal.md](journal.md), Laziness and approval against real ballots) suggest about 0.1 for an engaged electorate in a Condorcet election with few candidates (Debian) and 0.2–0.5 for public and organisational STV elections with many candidates (Irish STV elections, ERS data), so 0.1 and 0.5 are sensible values for a robustness check. Real ballots also have a few percent of bullet votes even where almost everyone else ranks fully, which one uniform cut cannot produce. IRV ballots are no guide: truncation there is partly strategic (lower preferences rarely count), which the strategic voters already model.
- **Honest ballot and strategy.** The lazy ballot is the voter's honest ballot: what non-strategic voters cast, round 0, and the base of VSE and WME. A strategic group's reference ballot (where its random search starts) uses the mean cut of its members. Strategic voters may rank more or fewer when it pays; kind voters only for a strict gain, so laziness persists unless strategy rewards completing the ranking.

**Two-round elections** (TTR, Approval runoff, Approval cond. runoff) have a real, separate runoff: voters come back and vote between the two finalists, independently of their first vote. With two candidates, voting for the preferred one is always best, so the runoff is decided by everyone's sincere preference between the finalists (turnout changes between the rounds are not modelled). The ballot carries that sincere ranking as a fixed part; strategy changes only the round-1 marks. Methods that read the runoff from the same ballot (Supplementary vote, STAR, the BTR methods) are automated runoffs, where strategic voters can also misrank the finalists.

### Score range

`--max-score m` sets the range of the score ballot to 0–m for every method that uses it (Score, MJ, MAV, STAR, BTR-score, MARS, IC-MARS, MARS-T, ARA, ARA-Top, Score+, Reciprocal score, Smith//Score, MMWV+Score, MMWVxScore, STLR, Soft Minimax, Soft Minimax-40, Soft Minimax-L, Soft Minimax-D, INS, Geometric median, and add-ons on score ballots). The default 0–5 is an implementation choice, not part of these methods; other simulations (e.g. Quinn's VSE) also run 0–10 and 0–100. With a coarse range, sincere voters must give candidates they like differently the same score, so pairwise preferences are lost: this is the cap on WME seen for the rated and hybrid methods. A wider range reduces it. MCA and 321 keep their three grades, and methods on other ballots are unaffected; the electorates are the same for every value, so a sweep (`sweep.py --max-score 5 10 100`) isolates the effect of the range, and its chain plot draws the reachable frontier.

Rules that refer to the range scale with it: ARA approves ratings above the middle, ARA-Top only the top score, Score+ stretches to the full range, Reciprocal score measures support as a share of the top score. Learning: see the score mutation in Strategy engine.

**Global range and fixed range.** `--max-score` is the global range: every method on score ballots follows it, unless the method fixes its own range with `settings`, e.g. `Method("score", d_sum, ("sum",), settings={"max_score": 10})`. While such a method's ballots are made, changed (strategy) and tallied, its own range is in force; afterwards the global one again. A method with a fixed range is named after it, `Name-X` (`Score-10`), and its rows still show the run's global `max_score`, so the name is what tells them apart. Avoid fixing the range for the base version of a method: a base method that follows `--max-score` stays comparable across runs, and a max-score sweep varies it like all the others (a fixed one stays the same at every value of the sweep). A test checks that every base method follows the global range and that fixed ones are named `Name-X`. A fixed method is equivalent to its base version at that global range: `Score-10` in a run at `--max-score 5` gives exactly the rows of `Score` at `--max-score 10`. No built-in method fixes its range at present. The settings a method can fix are listed in `SETTINGS` in methods.py (so far only `max_score`).

## Strategy engine

Two models of strategic voting (`strategy.py`); the front-runners, the groups answering around them and the attention to both sides (attack and defence) take inspiration from John Huang's votesim ([summary report](http://votesim.usa4r.org/summary-report.html), [code](https://github.com/johnh865/election_sim)), which measures one-sided strategy (only the challenger's coalition uses tactics) and two-sided strategy (the honest winner's supporters answer with a bullet vote). **Poll rounds** is the default; **convergent learning**, a long-run model kept for comparison, is available with `--converge N`.

### Poll rounds (default)

One election, with a few polls before it. Voters see only published poll results, never anyone's preferences, and answer them all at the same time.

- **Round 0:** everyone votes sincerely (naive, uninformed). `--rounds 0` reports this as the strategic result.
- **Round r** (1 … `--rounds`, default 3): a poll of round r−1 is published, and every strategic group answers it at once. Nobody sees anyone else's answer before the next poll. The strategic axes report the last round. Each round answers a pure poll of the round before (level-k reasoning: round 1 best-responds to sincere voters, round 2 to round-1 voters, …).
- **What a poll shows:** the method's own counts, the summed ballot features it decides from (first preferences for Plurality, approvals for Approval, the pairwise matrix for Condorcet methods, score totals and head-to-heads for STAR, …), from a random sample of `--poll-size` voters (default half the electorate, rounded to an odd number: 99 → 49, 100 → 51, 200 → 101), scaled up to the electorate. `--samples` independent polls (10) stand for the uncertainty. For some methods these counts say more than a real poll would (IRV's counts include every elimination order); methods with richer counts may profit from that.
- **Uncertainty:** voters treat a poll as `--poll-error` times (2) less precise than its sampling error: the polls are spread around the true count by that factor. Real polls miss results by about twice their stated sampling error (Shirani-Mehr, Rothschild, Goel & Gelman 2018). With the sampling error alone, the leader's supporters almost never see a poll in which they are behind, so they never protect their lead, while the runner-up's supporters attack it.
- **Groups:** whole ranking groups (strategic voters with the same sincere ranking, no cap). Voters who share a ranking and see the same poll reach the same answer, so they act as a bloc. Factions across rankings are not built in; they emerge when groups answer the same poll alike (all who prefer the runner-up to the leader, say).
- **Leaders:** the two candidates that win most often over the round's polls, the same for everyone. If one candidate wins every poll, the polls are widened (deviations × 2, 4, …; counts that would cross zero are clipped there) until another one wins somewhere: the race that would matter first.
- **Answer:** a group estimates the count for a ballot b as poll count − its own previous ballots + its size × b, and judges b by its expected utility gain over its sincere ballots, over the polls. It first tries a few candidate ballots: its previous answer, and the naive exaggerations of its ballot type around the two leaders (A = the one it prefers, B = the other):

  | Ballot | Naive exaggerations |
  |---|---|
  | one name | vote A |
  | approval | approve everyone liked at least as much as A |
  | score, three grades | top grade for everyone liked at least as much as A, bottom for everyone liked no more than B, stretched in between; or top down to A and bottom below |
  | ranking | A to the top and B to the bottom; A to the top; B to the bottom |
  | ranking with ties | everyone liked at least as much as A tied at the top, with or without B at the bottom; or the ranking moves |
  | truncated ranking | rank only down to A, sincerely or with A first |
  | ranking of at most 3 | A first and B never ranked; or rank only down to A (at most 3) |
  | two-round, plurality first round | mark A (the runoff ranking stays sincere) |
  | two-round, approval first round | approve everyone liked at least as much as A (the runoff ranking stays sincere) |
  | veto | veto B |

  **Aggressive** voters take the best of these if it does not lose in the polls: exaggeration wins ties, they exaggerate when it cannot hurt, whatever the gap. **Kind** voters take it only if it strictly gains: honesty wins ties. `--aggressive` (0.5) is the share of strategic voters who are aggressive; each voter's type is drawn per election from its own random stream (the same for every method), and aggressive and kind voters form separate groups. They are only candidates: one that backfires in the polls is not taken. Then random search (at most `--search` proposals, 10; it stops after 3 in a row without gain) takes a change only if it strictly gains. A group none of whose candidates is safe and whose search finds no gain stays sincere. Without a runner-up (exact polls) there are no exaggerations, only the search.
- **Settled:** the last round gave the same ballots as the round before (a fixed point).
- Condorcet-random-drop, which has no feature sums, is polled by tallying the sampled ballots, with the group's sampled members changed.

### Partial strategy

`--strategic p` makes a random share p of the voters always strategic; the others are adaptive (by default, see Adaptive voters below) or, with `--adaptive 0`, always sincere. Each voter is strategic with probability p, drawn from its own random stream per election, so the same voters are strategic for every method. Groups (and their size cap) form only among the strategic voters. p = 1 gives full strategy; p = 0 with no adaptive voters (`--adaptive 0`) gives the sincere results. A share like 0.5 with `--adaptive 0` (the default until 2026-10-17) is an intermediate level, in the spirit of other simulations that report 0, 50 and 100% strategy, and comparable with them. Sincerity and the strategy types are averaged over all voters, including the sincere ones.

Run time falls less than in proportion to p: halving the strategic voters removes fewer than half of the groups, since many groups are small.

**Adaptive voters** (`--adaptive a`, poll rounds only). A fixed share of strategic voters makes some of them optimise their ballot even when nothing is at stake for them, while sincere voters, lazy ones in particular, never react. Adaptive voters strategise when the race that matters matters to them.

- Three groups, from the same random draw per voter: always strategic (a share p, `--strategic`), adaptive (the next share a) and always sincere (the rest; p + a ≤ 1). With `--adaptive 0` the results are exactly those of the share model.
- **The default** (since 2026-10-17): `--strategic 0` and `--adaptive` left out, which means everyone not always strategic is adaptive (a = 1 − p; with `--converge`, a = 0, and a `--strategic` share must be given). So by default every voter is adaptive, and `--strategic 0.5` alone gives half always strategic, half adaptive; the share model needs `--adaptive 0`. Why: adaptive voters strategise when they have a reason, need no share to be chosen, and kept the method order of the share model while removing its artefacts (strategy without a stake, sincere voters who never react; [journal.md](journal.md), Adaptive voters).
- **Stake.** In each round, after the poll, the two leaders A and B are known (the same for everyone; Strategy engine). A voter's stake is |u(A) − u(B)|, rescaled to 0–1 between their least and most liked candidate: how much the outcome of the leading race matters to them.
- **Decision, every round anew.** Each adaptive voter has a personal threshold, fixed for the election: their place in the adaptive band of the draw, uniform on 0–1. In a round, the voter votes strategically if their stake exceeds it, and sincerely otherwise (lazy if `--lazy` is set). So the share strategic at a given stake equals that stake: a voter indifferent between the leaders never strategises, one with everything at stake always does. A voter can turn sincere again when the leaders change.
- **Both sides.** In a close race the supporters of both leaders have a stake, so attack and defence come from the same rule; the leader rule of Myerson and Weber needs both. A method whose two leaders are both broadly acceptable gives fewer voters a reason to strategise.
- The report gives the share of adaptive voters voting strategically in the reported round, per method (column `adapted` in the elections CSV).
- Not with `--converge` (convergent learning has no leading race per round); `sweep.py` skips such combinations, and those with p + a above 1. With `--start previous`, the previous election uses the same groups.

### Convergent learning (`--converge N`)

The earlier model: over N rounds, groups take turns and each sees the others' latest strategic ballots through a fresh poll of the full electorate. It finds where an electorate ends up after long adjustment, with much more information and coordination than one election offers. That makes it resistant to strategies that only fail when many voters use them at once: mass burial under Borda, for example, which sinks Borda when everyone buries simultaneously ([journal.md](journal.md)). Kept for comparison and as a long-run view; the results up to 2026-10-08 used it with N = 20 and 10 poll samples (`--converge 20 --samples 10`).

- **Groups:** strategic voters with the same sincere ranking, split at random into subgroups of at most `--group-size` voters (see Coordination, below). A group either votes sincerely (each member their own sincere ballot) or casts one shared ballot.
- **Polls:** each round draws `--samples` bootstrap resamples of the electorate. A proposal is judged by the group's total expected utility averaged over these samples, so a group gains in proportion to how often it would be pivotal, not only when it is decisive in the exact electorate.
- Voters start sincere. A group that has not deviated yet changes the sincere ballot of its average member (mean utilities of the group); a deviating group changes its current shared ballot, and sometimes (15% of proposals) proposes going back to sincere.
- **Acceptance:** a proposal is kept only if it raises the group's expected utility. A return to sincere is also kept when it is not worse: voters prefer honesty when indifferent. Without this, a deviation that once paid off stays forever once it stops mattering, and sincerity keeps falling with the number of rounds instead of reflecting what strategy actually pays.
- **Rounds:** groups in random order each round; no early stop. Each group tries only one random change per round, so a round without any accepted change does not mean learning has finished, and `settled` (no change in the last round) is a weak sign of convergence.

**Coordination: what the strategic axes measure** (convergent learning only; poll rounds always use whole ranking groups, and there coordination is limited by information, one poll and simultaneous answers, instead of by a group-size cap).

How many voters move together decides which strategies pay, and the methods differ sharply in this.

- **Compromise** (voting for a viable candidate instead of a hopeless favourite) pays even for a single voter. It is the main strategy in TTR (top-two runoff) and IRV, and it tends to carry the Condorcet winner into the final round.
- **Burial** (ranking or rating a rival below a candidate one likes less, to create a cycle or to drop the rival in a runoff) only pays when many voters do it together. A single voter out of 99 is almost never pivotal for it. It is the main strategy against Condorcet methods, STAR and BTR-score, and it targets exactly the Condorcet winner.

With `--group-size 0` (whole ranking groups) the strategic axes measure resistance to **organised factional strategy**: every voter with the same ranking follows one recommendation, the case most favourable to burial. With `--group-size 1` they measure resistance to **individual strategy**, where burial almost never pays. The default of 10 lies in between; results under strategy should always be read together with the group size used.

Groups are formed from voters with the same full sincere ranking. With many candidates there are many possible rankings (720 for 6 candidates), so most ranking groups are small whatever the cap, and organised blocs become rare: with 6 candidates and 99 voters, the group size changes little. Coordination by favourite or by cluster would be a different model.

A second, smaller asymmetry: learning is myopic. A burying group does not anticipate that the Condorcet winner's supporters may bury back, or that the cycle it creates can elect a candidate it likes even less (the chicken dilemma). This probably makes burial look somewhat more attractive than it would be to far-sighted voters.

Smaller groups mean more groups, and run time grows with their number.

### Common to both

- **Proposals** (one small change to a ballot; `strategy.mutate`):
  - one name: switch to another candidate
  - ranking: swap two candidates
  - approval: toggle one candidate (never empty or full)
  - two-round ballots: move the round-1 mark to another candidate, or toggle one approval (never empty or full); the runoff ranking never changes
  - veto: move the veto to another candidate
  - ranking with ties: move one candidate to any other level (equal levels are ties)
  - truncated ranking: swap two candidates, or rank one more or one fewer
  - score: set one candidate to a different value 0–max. For ranges above 0–5, half of the proposals use the six levels of the 0–5 scale stretched to the range (0, 20, … 100 for 0–100), so the chance of trying an extreme does not shrink as the range grows and learning is not slowed by the range alone
  - three grades: set one candidate to a different grade
- **Feature sums:** every tally except Condorcet-random-drop depends on the ballots only through sums over voters of per-voter features (totals, first and last places, the pairwise matrix, grade counts, first choices among each subset of candidates, ...). Each method lists its features and decides from their sums. In both models a group's change is evaluated by updating the polled sums instead of recounting. In convergent learning all features are integers, so this gives exactly the same ballots as recounting (checked against the earlier code); poll counts are scaled from a sample and need not be integers.
- **Random streams:** every method learns with a fresh copy of the same random stream per election (common random numbers: the same polls for every method as long as they draw alike), and the candidates' order of moves in strategic nomination is likewise the same for every method. So results do not change when methods are added or reordered, methods that agree on every ballot set agree in the results (IRV and ER-IRV on honest ballots), and paired differences are not blurred by different poll draws.

## Starting information

`--start previous` asks whether a method lets the electorate leave a strategic state it no longer agrees with. The first poll does not show the honest ballots but the result of a **previous election**: one extra (dummy) electorate under the same settings, with its own random draw, run through the same number of poll rounds under the same method. Its final ballots are taken as they are and its candidates relabelled at random as the current election's candidates (a new random assignment per election). Everything else is unchanged: the electorate, the honest results and all later polls are those of the current election, and voters start from their sincere ballots; only the information in round 1 differs. So a sweep `--start honest previous` compares the same elections, and the difference is what the outdated picture costs after 1, 3, 8, ... rounds: how strongly a method holds on to a strategic attractor rather than following the real electorate.

- The same previous election is used for every election of a run, so elections stay independent of each other (no drift from one to the next).
- It needs the original field, so it cannot be combined with strategic nomination, and it is for poll rounds only.
- Cost: one extra election per method and run.
- The previous electorate's candidates sit elsewhere in the issue space than the current ones, but only their counts carry over, relabelled; the current candidates and voters are untouched (a shifted opinion spectrum, with the same candidates, would be a different test).

## Strategic nomination

`--exits 1` and `--clones N` let the candidates of the initial draw change the field before the vote (`nomination.py`; after Green-Armytage 2014, "Strategic voting and nomination", *Social Choice and Welfare* 42, with clones added). Nobody new enters.

- **Moves.** Exit (at least two original candidates stay in), re-entry after an exit, fielding a clone of any original in the race (itself included; at most one clone per original and N in the race), withdrawing a clone they fielded. A clone is a copy at a small random offset (0.01 per dimension), so voters rank it next to its original and split between the two.
- **Who moves and why.** Candidates sit in the issue space and value a winner like a voter at their position (exp(−distance/width); themselves 1). A clone counts exactly as its original. A candidate makes a move only if it strictly improves their expected utility under the method's result with **sincere** ballots on the new field: candidates foresee honest voting, not learned strategy.
- **Dynamics.** Rounds in which the candidates take turns in random order, each making its best improving move. A round without a move ends it (an equilibrium); otherwise it stops after 20 rounds. Each method gets its own field. The voters then vote on that field, honestly and with learned ballots as usual.
- **Baseline.** Results are judged on the **original field**: a clone's chance of winning counts as its original's, and withdrawn candidates stay in the comparison. VSE, WME, the random ballot, the best candidate and the reachable frontier are therefore the same as without nomination, and the change in VSE and WME is what nomination does to the outcome. Withdrawing the best candidate counts as a loss.
- **Report.** With nomination on, each condition gets a table per method: how often the honest result changed (Green-Armytage's measure of vulnerability, here after the dynamics rather than for a single move), the average number of exits and clones at the end, and how often the field settled. A sweep `--exits 0 1` or `--clones 0 2` draws chain plots with the frontier.

Cost: each candidate move is checked with one sincere tally on the changed field, a few hundred tallies per method and election; learning remains the larger part.

## Participation

`--participation` checks the no-show paradox (`participation.py`): could some voters get a better result by staying home? It is a diagnostic; nobody actually abstains, and the strategic results do not change (staying home is not among the strategic voters' options).

- **Who stays home.** Voters with the same sincere ranking and the same ballot form a group. For each group, 1, half (rounded up) or all of its members leave out their ballots, while everyone else votes as before. A single voter is rarely pivotal among 99, so the larger shares matter.
- **Failure.** The resulting winner lottery is better for those who stayed home (expected utility, summed over them) than the one with their ballots. Leaving out a ballot means the whole ballot, in a two-round election both rounds.
- **Both ballot sets.** Honest: everyone votes sincerely (the classical participation criterion, for groups). Strategic: everyone keeps the learned ballots and the group drops its learned ballot; a strategic ballot that backfires shows up here.
- **Report** (when the CSV has the columns): per method, the share of elections with a failure (with a 95% interval), the share of voters in failing groups, and where it fails, the mean of the largest gain per absent voter as a share of that voter's utility range (favourite minus least liked).
- Score, Approval and Borda satisfy participation with sincere ballots, so they fail only with learned ones; IRV, TTR and the Condorcet methods (with 4 or more candidates) can fail with sincere ballots too.

Cost: one tally per group and share (feature sums minus the absent ballots), a few hundred per method and election; with all methods at the defaults the run takes about 5% longer.

## Manipulation test

`--manipulation` asks how bad deliberate, informed manipulation from honest ballots can get (`manipulation.py`), after John Huang's votesim ([summary report](http://votesim.usa4r.org/summary-report.html)) and the vulnerability measure of Green-Armytage, Tideman & Cosman (2016, "Statistical evaluation of voting rules", *Social Choice and Welfare* 46). It is a diagnostic beside the poll rounds: the poll rounds show what typical strategy does, this test the worst a coalition with perfect information can do. It changes no other result.

- **Coalitions.** T is the honest winner (the most likely one if the honest result is a lottery). For every other candidate X, the coalition is every voter who prefers X to T.
- **Tactics.** The coalition tries each naive exaggeration of its ballot type for X against T, the same ones the poll rounds start from (X to the top and T to the bottom of a ranking, approve everyone liked at least as much as X, top score down to X and 0 from T down, …; see Strategy engine). Every member uses the same tactic on their own utilities; a member whose ballot type offers none keeps their sincere ballot. Everyone else votes sincerely.
- **One-sided.** A tactic succeeds if the coalition gains (expected utility summed over its members). Recorded: whether some coalition succeeds, and the total utility of the successful manipulation that is worst for the electorate; without a success, the honest result.
- **Two-sided.** For each successful tactic, T's supporters (everyone not in the coalition) answer with an exaggeration for T against X: of their ballot type's naive exaggerations, the one that leaves the coalition the least gain, so a manipulation stands only if it pays against every answer. Recorded the same way, among the manipulations that stand. votesim lets the defence bullet-vote and keeps backfired two-sided outcomes too; here the defence tries every exaggeration of its ballot type and only manipulations that still pay count.
- **Report.** Per method: the share of elections in which some coalition can manipulate, one-sided and two-sided (with 95% intervals), and the VSE of the recorded outcomes (ratio of sums, as VSE). Columns `manip1`, `manip2`, `U_manip1`, `U_manip2` in the sincere rows of the elections CSV.
- With strategic nomination, the test runs on the field the voters face and is judged on the original field, like everything else.

Cost: a few tallies per candidate and tactic, a few dozen per method and election; with the standard set a run takes only slightly longer.

