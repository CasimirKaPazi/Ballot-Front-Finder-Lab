# Methods

The voting methods, their groups and how to add your own. How each ballot type is filled in sincerely is in [model.md](model.md), Ballots and sincere voting.

## Groups and sets

Every method belongs to exactly one group. Groups can be used wherever methods are named (`--methods`, `--exclude` in run.py and sweep.py; `--show`, `--hide`, `--track` in analyze.py), they are the panels of the chain plots, and the report and summary.csv show each method's group.

| Group | Methods |
|---|---|
| `condorcet` | Ranked pairs, Schulze, Schulze-ER, Kemeny-Young, Copeland, Copeland//Margins, Copeland//Margins-ER, SICT (Condorcet methods decided by the pairwise counts alone) |
| `minimax-family` | Minimax, Minimax-ER, MMWV, MMPO, MMPOWV (the minimax family; MMPO and MMPOWV are not Condorcet methods) |
| `condorcet-based` | Black, Baldwin, Baldwin-ER, Nanson, BTR-Borda, BTR-IRV, Benham, Smith//Score (Condorcet methods built on a Borda, IRV or score count) |
| `plurality-based` | Plurality, Supplementary vote, IRV, ER-IRV, IRV-3, Coombs, DSC, DAC (first choices, eliminations and coalitions of top preferences; Coombs eliminates by last places) |
| `approval-based` | Approval, MDDA, MAMPO |
| `rated` | Score, Reciprocal score, INS, DMS, ESR (add-ons can join groups, see Add-ons) |
| `median-based` | MJ, MAV, MCA (the grade a candidate reaches with a majority: median grade or Bucklin-like) |
| `hybrid` | STAR, BTR-score, MARS, IC-MARS, MARS-T, ARA, ARA-Top, Score+, 321, MMWV+Score, MMWVxScore, STLR, Soft Minimax, Soft Minimax-40, Soft Minimax-L, Soft Minimax-D (ratings combined with pairwise comparisons or runoffs) |
| `positional` | Borda, Borda-ER, Dowdall |
| `two-round` | TTR, Approval runoff, Approval cond. runoff, Score runoff (a real, separate runoff; see below) |
| `references` | Random pair runoff, Anti-plurality, TRASH, Geometric median, Condorcet-random-drop, CW else Score loser, CL else Score winner (benchmarks and idealised rules, not proposals; not run unless named, e.g. `--methods all` or `--methods references STAR`, and hidden in the plots by default) |
| `inverse` | Inverse-Plurality, Inverse-Approval, Inverse-Minimax, Inverse-IRV, Inverse-ER-IRV, Inverse-TTR, Inverse-Borda, Inverse-Score (tests of the strategy model, below; not run unless named; hidden in the plots by default) |

**Sets** are selection names across groups (not plot panels):

- `standard`, the default selection: the methods most discussed, those of Quinn's vse-sim (Plurality, IRV, Borda, Approval, Score, STAR, MJ, MAV, 321, Ranked pairs, Schulze) the common single-winner proposals and methods in use (TTR, Supplementary vote, ER-IRV, IRV-3, Approval runoff, Approval cond. runoff, Copeland, Kemeny-Young, Minimax, MMWV, MMPO, BTR-IRV), a reference between Score and STAR (Score runoff: STAR's runoff as a separate election, joined 2026-10-20), with the strongest found here (MARS, ARA, BTR-score, Smith//Score, Soft Minimax-D): 29 methods. Copeland//Margins-ER (the earlier Ranked Robin) left the set on 2026-10-19: once Equal Vote's name came to mean any Condorcet method, this particular variant is no longer a live proposal. Geometric median and Condorcet-random-drop are left out: idealised reference cases rather than practical methods.
- `extended`: the standard set and the strongest newer methods not in it (ESR, DMS, Soft Minimax, Soft Minimax-40, MARS-T): 34 methods. Geometric median was in it until 2026-10-26, when it moved to the references. The selection of the reference runs ([reference-runs.md](reference-runs.md)), so that every test shows them in all conditions.
- `max-score`: every method whose results depend on the score range (`--max-score`): those on a ballot type that uses the range (the score ballot, Score runoff's ballot, add-on types built on them) and that do not fix their own range. Built from the ballot types, so new methods and add-ons join by themselves; a test checks that it holds exactly the methods whose sincere ballots change with the range. MCA and 321 (three fixed grades) are not in it. It includes Inverse-Score and the reference methods on score ballots: `--methods max-score --exclude inverse references` for a max-score sweep.
- `er-pairs`, each strict method next to its equal-ranking version: IRV/ER-IRV, Borda/Borda-ER, Baldwin/Baldwin-ER, Minimax/Minimax-ER and MMWV, Schulze/Schulze-ER, Copeland//Margins/Copeland//Margins-ER. Equal-ranking versions equal the strict ones with honest ballots (sincere ballots are strict), so these pairs differ only in what strategic voters do with ties.

Example: `--methods condorcet STAR`, `--methods er-pairs`, `--methods max-score`, `--methods all --exclude inverse`. Without `--methods`, run.py and sweep.py run the `standard` set; `--exclude` then takes methods out of it. Every other method, group, add-on, or `all` can be named in `--methods`.

## Methods

75 built-in methods, listed by group (see Groups and sets, above), plus any add-ons (see Add-ons, below); a run uses the `standard` set unless `--methods` names others. "0–5" in the Ballot column is the score ballot with the default range; `--max-score` changes it for all of these methods at once.

Pairwise preferences always come from the ballot itself, so equal ratings count as no preference. All ties are broken uniformly among the tied candidates and counted exactly.

**condorcet**

| Method | Ballot | Rule |
|---|---|---|
| Ranked pairs | ranking | lock pairwise wins from the largest margin down, skipping any that would close a cycle; the candidate with no locked defeat wins. Equal margins are locked in every order and averaged (at most 240 orders, otherwise 240 fixed random ones) |
| Schulze | ranking | strongest path by margins; a candidate whose path to every rival is at least as strong as the rival's path back wins |
| Schulze-ER | ranking, ties allowed | Schulze on rankings with ties (equal-ranked candidates count as neither above the other) |
| Kemeny-Young | ranking | the ranking agreeing with the most pairwise preferences; its top candidate wins (exact, by dynamic programming over subsets); candidates heading different optimal rankings split |
| Copeland | ranking | most pairwise wins (ties count half) |
| Copeland//Margins | ranking | most pairwise wins (Copeland); ties go to the greatest sum of margins against the other tied candidates, then against all candidates; remaining ties are split. Equal Vote's earlier definition of Ranked Robin; the lab's name for it until 2026-10-19, when Equal Vote's "Ranked Robin" had come to mean any method electing the Condorcet winner, with a tiebreaker of the jurisdiction's choice (Copeland, most first places or Smith-Minimax; [equal.vote](https://www.equal.vote/ranked_robin)) |
| Copeland//Margins-ER | ranking, ties allowed | Copeland//Margins on rankings with ties (equal ranks: no preference). Sincere ballots are strict; ties are learned. In the standard set until 2026-10-19 |
| SICT | ranking, ties allowed | Symmetrical ICT: p(X,Y) if (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top); X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins; none or all unbeaten: most first places; otherwise most first places among the unbeaten. Sincere ballots are strict; ties are learned |

**minimax-family**

| Method | Ballot | Rule |
|---|---|---|
| Minimax | ranking | smallest worst pairwise defeat (by margin) |
| Minimax-ER | ranking, ties allowed | Minimax on rankings with ties (equal-ranked candidates count as neither above the other) |
| MMWV | ranking, ties allowed | Minimax by winning votes: the worst defeat is measured by the votes for the winner of that pair. On strict rankings identical to Minimax (margins); differs only once equal rankings are learned |
| MMPO | ranking, ties allowed | minimax pairwise opposition: the smallest largest number of voters ranking some rival above the candidate, whether or not that rival wins the pair. Not a Condorcet method; on strict rankings identical to Minimax |
| MMPOWV | ranking, ties allowed | proposed in this lab: (1 − largest pairwise opposition / n) × (1 − worst defeat in winning votes / n); the highest product wins. Multiplying makes a weakness on either costly and is smooth, without thresholds for strategy to aim at. On strict rankings it orders candidates like Minimax, so it equals Minimax with honest ballots and differs only once equal rankings are learned. Not Condorcet-consistent once ties are used |

**condorcet-based**

| Method | Ballot | Rule |
|---|---|---|
| Black | ranking | Condorcet winner if there is one, otherwise the Borda winner |
| Baldwin | ranking | eliminate lowest Borda score, recounted among remaining |
| Baldwin-ER | ranking, ties allowed | Baldwin on rankings with ties (equal-ranked candidates count as neither above the other) |
| Nanson | ranking | repeatedly eliminate every candidate whose Borda score among the remaining is at or below the average (Baldwin eliminates only the lowest) |
| BTR-Borda | ranking | order by Borda score, bottom-two pairwise runoffs climbing up; a pairwise tie goes to the candidate higher in the Borda order (since 1.1.0; before, split evenly) |
| BTR-IRV | ranking | each round the two with the fewest first choices meet pairwise; the loser is eliminated |
| Benham | ranking | IRV, but a Condorcet winner among the remaining candidates wins at once |
| Smith//Score | 0–5 | highest score total among the Smith set (pairwise from the ratings, equal ratings no preference). The same as BTR-score unless the Smith set has 4 or more members ([journal](journal.md), BTR-score and Smith//Score) |

**plurality-based**

| Method | Ballot | Rule |
|---|---|---|
| Plurality | one name | most first choices |
| Supplementary vote | ranking (first two used) | top two by first preferences; a ballot whose first preference is out counts for its second preference if that is a finalist, otherwise it is exhausted |
| IRV | ranking | eliminate fewest first choices among remaining |
| ER-IRV | ranking, ties allowed | IRV where every candidate tied at the top of a ballot (among the remaining) gets a whole vote, as in approval. Sincere ballots are strict, so it equals IRV with honest ballots; ties are learned |
| IRV-3 | ranking of at most 3 | IRV as used in practice with a limited ballot: at most 3 candidates ranked; a ballot whose ranked candidates are all out is exhausted and counts for nobody. Equals IRV with 3 candidates |
| DSC | ranking, ties allowed | Descending Solid Coalitions (Woodall): every set of candidates gets the number of voters who rank all of it strictly above all other candidates; from the strongest set down, each makes the candidates outside it ineligible unless none would remain; the last eligible candidate wins. Equal-strength sets whose order matters are applied in every order with equal weight. Clone-independent; with strict rankings the coalitions are each voter's top-k sets |
| DAC | ranking, ties allowed | Descending Acquiescing Coalitions: as DSC, but a voter counts for every set none of whose outside candidates they rank strictly above one inside (equal ranks count for both sides). Equals DSC on strict rankings; differs once voters tie candidates |
| Coombs | ranking | a candidate with more than half of the first choices among the remaining wins; otherwise the one with the most last places among the remaining is eliminated |

**approval-based**

| Method | Ballot | Rule |
|---|---|---|
| Approval | approval | most approvals; sincere: above the voter's personal threshold between mean and favourite ([model.md](model.md), Honest approval) |
| MDDA | truncated ranking (ranked = approved) | disqualify every candidate that more than half the voters rank some other candidate above (unless all would be); most approved remaining wins. Sincere: rank the candidates the voter would approve ([model.md](model.md), Honest approval) |
| MAMPO | truncated ranking | Majority Approval, Minimum Pairwise Opposition (Kevin Venzke): ranked = approved. If nobody is approved by more than half the voters, the most approved wins; otherwise, among those approved by more than half, the one with the smallest largest pairwise opposition (most voters ranking some rival above it) wins. Devised to show that favourite betrayal, strong defensive strategy and strategy-freeness are compatible |

**rated**

| Method | Ballot | Rule |
|---|---|---|
| Score | 0–5 | highest total; sincere: stretched to the voter's range |
| Reciprocal score | 0–5 | interpretation of [electowiki](https://electowiki.org/wiki/Reciprocal_Score_Voting): a voter's faction is their top-rated candidate(s). R(X→Y) is the mean rating supporters of X give Y, as a share of the top score. A score for c is multiplied by min(1, R(c→F)/R(F→c)), F the voter's faction: support that c's supporters do not return is cut. Scores for the voter's own top and for candidates nobody top-rates stay unchanged; highest total wins |
| INS | 0–5 | iterated normalized score, a score elimination: each round every ballot is stretched to the full range over the remaining candidates (lowest 0, highest the top score; a ballot rating them all the same counts for none), the lowest total is eliminated (ties for last split the win), until one is left |
| DMS | 0–5 | depth maximin (proposed in this lab): the ballots are read as levels from the top score down (score 0 is no level). For each depth k, the top k levels, a candidate's value is the points they get from those levels divided by the points of the levels counted (5, 5+4, 5+4+3, …); the candidate whose lowest value over the depths is highest wins, ties by the next lowest (leximin), still tied: split. A candidate must be strong at every depth: a dark horse with no top scores has 0 at depth 1, a polarizing one falls at the deep levels. Mostly agrees with Score; where they differ, a middle depth usually decides. Variants tried (relative values, other weights, a soft minimum, elimination, a Buck-set runoff, and others) were not better |
| ESR | 0–5 | expanding support runoff (proposed in this lab): the ballots read as levels from the top score down, as for DMS. For every depth k (the top k levels) the candidates with the most score points from those levels join the set (all tied for most); at the full depth that is the score total, so the Score winner is always in it. A set of one wins outright; otherwise every ballot is stretched to the full range over the set (not rounded; a ballot rating the whole set alike counts for nobody) and the highest total wins. Tested as BSR-S (a variant counting voters instead of points, a Buck-set runoff, was weaker and is not in the lab); results stored under that name are read as ESR |

**median-based**

The grade a candidate reaches with a majority of the voters: the median grade, or Bucklin-like, the first level from the top at which someone is rated that high by more than half.

| Method | Ballot | Rule |
|---|---|---|
| MJ | 0–5 | majority judgment: highest median grade; ties broken by repeatedly removing the median |
| MAV | 0–5 | Majority Approval Voting (graded Bucklin; [electowiki](https://electowiki.miraheze.org/wiki/Majority_Approval_Voting)): grades are counted from the top down; at the first grade where some candidate is graded that high or higher by a majority, the candidate with a majority and the most votes at the grade above wins (at the top grade: the largest majority). Unlike Bucklin, a larger majority at the lower grade does not decide; vse-sim uses five grades and a fractional median score instead |
| MCA | three grades | majority choice approval, MCA-A variant: the first grade level (good, then good or OK, then all) at which some candidate is rated that high by more than half the voters decides; one such candidate wins, among several the most approved (good or OK) |

**hybrid**

| Method | Ballot | Rule |
|---|---|---|
| STAR | 0–5 | top two by total, majority runoff; a runoff tie goes to the higher total |
| BTR-score | 0–5 | order by total score, bottom-two pairwise runoffs climbing up; a pairwise tie goes to the candidate higher in the score order (since 1.1.0; before, split evenly) |
| MARS | 0–5 | as BTR-score; each match adds share preferring to score as share of maximum |
| IC-MARS | 0–5 | MARS where a pairwise margin counts only beyond the votes rating both candidates at the top (tied at the top, as in [Improved Condorcet Approval](https://electowiki.org/wiki/Improved_Condorcet_Approval)): a margin those votes could reverse counts 0, so score weighs more |
| MARS-T | 0–5 | MARS with each pairwise margin scaled by the share of ballots that score the two candidates differently (1 − share of equal scores): pairs that many voters do not tell apart are decided more by the score difference |
| ARA | 0–5 | ratings above the middle of the range are approvals (0–5: 3–5; 0–10: 6–10); order by approvals; each match adds voters preferring to approvals (formerly MARA) |
| ARA-Top | 0–5 | as ARA, but only the top score counts as approval |
| Score+ | 0–5 | simplified variant from the [votingtheory.org thread](https://www.votingtheory.org/forum/topic/138/score-interval-score-with-additional-protection-against-the-chicken-dilemma): A defeats B robustly if A's total minus the scores both share (per voter the lower of the two) still exceeds B's total. Robustly defeated candidates are eliminated, each voter's scores among the rest are stretched again to the full range, highest total wins |
| 321 | three grades | 3-2-1 voting: 3 semifinalists with the most "good" (the third only with at least half as many as the first), of them 2 finalists with the fewest "bad", then the finalist rated above the other on more ballots. Ties: higher score (good 2, OK 1), then split. The party rule for semifinalists does not apply |
| MMWV+Score | 0–5 | score share (total ÷ (voters × top score)) minus the worst defeat in winning votes as a share of voters; the highest wins (proposed in this lab; pairwise from the ratings) |
| MMWVxScore | 0–5 | score share × (1 − worst winning-votes defeat share), as in MMPOWV |
| STLR | 0–5 | score then levelled runoff: the two highest score totals go on; in the runoff each ballot's scores for the two are rescaled so that the preferred one gets the top score (the other keeps its ratio, e.g. 2 and 1 become 5 and 2.5); a ballot giving both 0 counts for neither; a runoff tie goes to the higher total |
| Soft Minimax | 0–5 | proposed in this lab: Minimax on soft margins. For each pair, every ballot's score difference is clipped to ±20% of the range and divided by that, so a difference of a fifth of the range or more is a full vote for one side and smaller ones count in proportion; the candidate whose worst soft margin is best wins. With the default 0–5 scale the clip is 1 point, so on integer scores this is exactly Minimax by margins on the score ballots; with a wider range (`--max-score 10`) small differences count in part. Clipping at the whole range would give Score |
| Soft Minimax-40 | 0–5 | the same with the clip at 40% of the range (2 points on 0–5): 1-point differences count half; between Soft Minimax and Score |
| Soft Minimax-L | 0–5 | Soft Minimax with an adaptive clip per pair: the mean of the two candidates' average scores (A averages 3, B 2: clip 2.5 points). Pairs of well-liked candidates are compared more by score difference, pairs of little-liked ones more by majority vote |
| Soft Minimax-D | 0–5 | Soft Minimax with an adaptive clip per pair: 1 point plus the difference of the two candidates' average scores. Pairs close in average score are compared by majority vote, pairs far apart more by score difference |

**positional**

| Method | Ballot | Rule |
|---|---|---|
| Borda | ranking | points by position |
| Borda-ER | ranking, ties allowed | a point for every candidate ranked strictly below; ties score nothing. Equals Borda on strict rankings |
| Dowdall | ranking | positional, 1, 1/2, 1/3, … points by rank (Nauru) |

**two-round**

A real, separate runoff: voters come back and vote between the finalists, by their sincere preference (see [model.md](model.md), Two-round elections). Not directly comparable with the one-ballot methods: the runoff is assumed sincere and turnout unchanged, and a second election has costs of its own.

| Method | Ballot | Rule |
|---|---|---|
| TTR | two-round, plurality first round | a candidate marked by more than half the voters wins; otherwise the top two go to a separate runoff, decided by sincere preferences |
| Approval runoff | two-round, approval first round | top two by approvals go to a separate runoff, decided by sincere preferences; a runoff tie goes to more approvals |
| Approval cond. runoff | two-round, approval first round | as Approval runoff, but a candidate approved by more than half the voters wins outright |
| Score runoff | two-round, score first round (0–5) | top two by score total go to a separate runoff, decided by sincere preferences (see [model.md](model.md), Two-round elections); a runoff tie goes to the higher total. STAR reads the runoff from the score ballot instead, so equal scores count for neither finalist and strategic voters can misrank them there; the difference between the two is what the automatic runoff costs or gains |

**references**

| Method | Ballot | Rule |
|---|---|---|
| Random pair runoff | ranking | two random ballots; same first choice wins, else majority runoff between the two |
| Anti-plurality | veto (one name against) | fewest vetoes wins; sincere: against the least liked |
| TRASH | ranking | "This Random Acronym Sounds Honest": the plurality runner-up and the anti-plurality runner-up (second fewest last places) meet in a runoff, and the one ranked lower by more voters wins; if both are the same candidate, it wins. A deliberately bad reference |
| Geometric median | 0–5 | the geometric median of the score ballots, taken as vectors over the candidates (the point with the smallest total Euclidean distance to all ballots; at most 20 Weiszfeld iterations from the mean); the candidate with the highest coordinate wins. The mean would be Score, the coordinate-wise median a median grade. Each ballot pulls with the same force however far away it is, so exaggerating in the same direction adds no pull. No feature sums, so polls use smaller samples, as for Condorcet-random-drop |
| Condorcet-random-drop | ranking | Condorcet winner if there is one; otherwise voters are dropped one at a time in random order until the rest have one. Averaged over 32 fixed random orders (the same on every call), so probabilities come in steps of 1/32 |
| CW else Score loser | 0–5 | the Condorcet winner of the score ballots (equal scores: no preference) if there is one, otherwise the candidate with the lowest score total. A test of the measures: a Condorcet method where the ballots show a Condorcet winner, as bad as possible where they do not. On 0–5 score ballots equal scores hide the Condorcet winner in some elections (13% at the defaults), so its CE is below 100 |
| CL else Score winner | 0–5 | the Condorcet loser of the score ballots (beaten by every other candidate) if there is one, otherwise the Score winner. A test of the measures. The spatial electorates nearly always have a Condorcet loser (99% at the defaults), so it is mostly the Condorcet loser, far below random ballot on both axes; Score only where the ballots show none |

**inverse**: tests of the strategy model. Each has the ballot and the count of its original, but its sincere ballot expresses reversed preferences, so honest voting elects (nearly) the worst candidate. Voters who learn to reverse their ballot get the original method back; how far the poll rounds get there shows how well the strategy model finds strategies that are not small steps from the honest ballot.

| Method | Ballot | Rule |
|---|---|---|
| Inverse-Plurality | one name | most named wins; the sincere ballot names the least liked candidate |
| Inverse-Approval | approve or not | most approvals win; the sincere ballot is the approval rule applied to the reversed preferences (the least liked always, others below the voter's mean by chance) |
| Inverse-Minimax | ranking | Minimax; the sincere ballot ranks upside down, so honestly the Condorcet loser (if any) wins |
| Inverse-IRV | ranking | IRV; the sincere ballot ranks upside down |
| Inverse-ER-IRV | ranking, ties allowed | ER-IRV; the sincere ballot ranks upside down (strictly, like every sincere ballot); strategic voters can tie candidates |
| Inverse-TTR | two-round, plurality first round | round 1 as TTR, the sincere ballot marks the least liked. The runoff is a separate two-candidate vote in which voters vote by their real preference whatever the ballot means: the fixed runoff part of the ballot is the reversed ranking and the count reads it back, so the real majority decides. Only round 1 can be learned, as in TTR |
| Inverse-Borda | ranking | Borda; the sincere ballot ranks upside down |
| Inverse-Score | 0–5 | Score; the sincere ballot scores upside down (the least liked gets the top score) |

## Add-ons: methods in separate files

A new method does not have to go into methods.py: put a Python file into the folder `addons/`, and its methods are loaded at start and used like the built-in ones, by name or group, in run.py, sweep.py and analyze.py. Delete the file to remove them; to share a method, share the file. Files starting with `_` are not loaded. `addons/_template.py` is a commented starting point; `addons/ids.py` (IDS, iterated dropped score) is a working example. [addons.md](addons.md) is a tutorial for writing one with little Python: a line-by-line walkthrough of IDS, and a new method (Bucklin) built step by step. A file defines:

**An add-on is a Python program.** Every file in `addons/` is run with your user's rights whenever the lab starts (run.py, sweep.py, analyze.py, the tests), and it can do anything a program can: read and write your files, use the network. Add only files whose source you trust, and read a shared add-on before putting it into `addons/`. The lab cannot check this for you.

| Name | Required | Meaning |
|---|---|---|
| `METHODS` | yes | `{name: Method(ballot, decide, feats)}` (or `Method(ballot, full=tally)` for a rule that needs the whole ballot matrix); names must not clash with existing ones. Optional `settings={"max_score": X}` fixes the score range for this method, named `Name-X` (see [model.md](model.md), Score range) |
| `FEATURES` | no | new per-voter ballot features, `{name: (function(B) -> array per voter, shape of the sum given C)}`; they are summed over voters and poll samples, which keeps tallies fast |
| `BALLOTS` | no | new ballot types, `{name: ballots.Ballot(...)}`: honest ballots, strategic changes, exaggerations, implied values and the worst ballot (see ballots.py; rarely needed). A type that uses the score range sets `score_range=True` (one made with `dataclasses.replace` from the score ballot keeps it), so its methods join the `max-score` set |
| `GROUP` | no | the group its methods join: an existing one (e.g. `rated`) or a new one; default `addons` |
| `test()` | no | its own checks (a small profile with a known winner), run by test_methods.py |

The ballot type (one of the built-in ones in ballots.py) decides the sincere ballots, laziness, the strategic moves and the measures, so a new method gets all of these for free. Results of add-on methods can be combined with existing runs like any other method ([running.md](running.md), Combining runs): run only the new method with the same parameters and seed.

**Current add-ons:**

| Method | Ballot | File | Description |
|---|---|---|---|
| IDS | 0–5 | `addons/ids.py` | iterated dropped score, a score elimination: each round every ballot's lowest-rated remaining candidates are struck out (count 0; a ballot rating them all the same counts for none), the other scores stay as given; the lowest total is eliminated (ties for last split the win), until one is left. Unlike INS a ballot only changes where its own lowest remaining rating is, and it can be counted by hand. Group: rated |
