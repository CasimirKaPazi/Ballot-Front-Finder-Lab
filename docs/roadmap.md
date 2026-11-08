# Roadmap

What is planned or under consideration for later versions of the lab, grouped by area. Each item says what it is, why it matters and where it came up. Items that change existing results need a minor or major version ([README](../README.md), Versions); the others can come in a patch. The order within a group is no priority.

When an item is done, it moves out of this file into the [journal](journal.md), with the date and what it changed.

## Strategy

- **Review the strategy model with Approval in mind.** Under the default conditions Approval hardly moves from its honest result (base: 86.3 / 90.4 honest, 86.3 / 90.3 strategic). Strategic approval voting is usually expected to move it towards the Condorcet winner, even with only part of the voters strategic. Either the model misses moves that real approval voters make (approving a second candidate against a front-runner they dislike, or withholding approval from a rival of their favourite), or the expectation does not hold in these electorates. *Findings 2.1.0, section 11.* Changes results.
- **Strategy without coordination: bounded rationality.** With coordination 0 nobody votes strategically, since a single voter's move never pays its cost (findings 2.1.0, section 5.2). Real voters vote tactically anyway, also when their vote cannot be expected to matter: some naive burial on ranked ballots, compromise out of habit. As in the beauty contest game, people with limited time for thinking act partly irrationally, and a model of strategy needs some of that. Possible forms: a share of voters who make the obvious move for their type whatever it is worth, or a lower cost for moves that are easy to think of. Needs care, since any such rule decides how much each method is hurt. Changes results.
- **Campaigns with pooled discovery.** Campaigns now need the arithmetic response (`--response arithmetic`), which misjudges pairwise counts (findings 2.1.0, sections 5.8 and 10.2). With pooled discovery, campaigns could be tested against the default engine. Once they are, the arithmetic response could be kept for comparison only. Changes results of campaign runs.
- **Smarter campaigns.** Campaigns that plan with the method's count, or across rounds, would have more effect than the current ones; the campaigns test showed little effect except under Plurality. Changes results of campaign runs.
- **Learning across elections.** Strategy now starts from a sincere poll, or from a previous election's result in the first-poll test. Repeated elections with the same electorate, where voters remember what worked, are not modelled (findings 2.1.0, section 10.5).

## Candidates and Nomination

- **Reference rules without strategic nomination.** The reference rules go through nomination like every method. Under the random ballot every candidate gains by fielding clones, so the reference rule Random ballot scores VSE −11 rather than 0 on the original field (findings 2.1.0, section 4.7). Exempting the references keeps the zero point of the scale exact. Changes the references' results only.
- **Costly clones.** A clone costs its party nothing in the model, while a real one needs a second credible candidate and splits a campaign; the no-nomination test may overstate what clones do (findings 2.1.0, section 10.1). A cost for fielding a clone, like the honesty cost for voters, would make nomination more realistic. Changes results.
- **Candidate fields closer to real ones.** In real data a few serious candidates are graded low by about half of the voters and the others by most of them: a cliff, while the model's candidates spread evenly (journal, approval data). Most voters probably have several candidates at the bottom of their scale, which is where rated methods are hit by nomination (journal, 2026-11-06). Better fields need data; candidate dispersion and shift are the current stand-ins.

## Methods and Compositions

- **More intermediate steps as parts**, so that methods can be compared on what they share, independently of their tie-breakers. Since 2026-11-08 the lab has the Schwartz, CDTT and Banks sets, IMDS's steps and MJ's median ([compositions.md](compositions.md)). Still missing, for example: the uncovered set; MJ's tie-break and MAV's grade-by-grade count as parts; DMS's depths.
- **A brute-force reference method.** Take the winners of all methods as a new field and hold a new election among them under all methods; repeat as long as the set changes. When it no longer changes, drop the candidates with the fewest wins (or break the tie otherwise, or by random ballot). Where the methods agree, this finds what they agree on; as a reference it shows how far a single method is from the consensus of all.
- **Multi-winner results.** Compositions that end in a set of k winners (committees) with the same machinery ([compositions.md](compositions.md), Next).

## Ballots

- **A realistic finer score range.** The score range changes the results across much of the field: on 0–5 the score–pairwise hybrids lose part of the voters' pairwise information to ties, on 0–99 several of them match the Condorcet methods (findings 2.1.0, section 6.2). 0–99 is impractical on a real ballot, 0–9 or 0–10 is not. The next reference runs should include 0–9 again, and the findings should state the score range's role wherever score methods are compared. Possibly a reason to reconsider the default range. Changing the default changes results.

## Measures and Analysis

- **Simplicity.** The front does not measure how simple a method is for voters, officials and the public (findings 2.1.0, section 10.4). A measure, even a rough one (steps of the count, whether it needs the ballots more than once, whether it can be counted by hand at polling stations and summed), would let simplicity enter the comparison instead of being weighed separately.
- **`--show` and `--hide` in general.** Since 2026-11-08 `--hide` applies within `--show`. Method sets that contain references are a general trap: the plots could leave out the references by default whatever selects the methods, and the report could say which methods each plot shows.
- **Identical results across machines.** The regression check can differ between machines where rounding decides a near-tie (another BLAS kernel; [regression.py](../regression.py)). Rounding the values compared in a voter's choice before the comparison would make results identical everywhere, at the cost of changing a few existing ones.

## Reference Runs and Validation

- **Test Soft Minimax-D outside this model.** Soft Minimax-D was proposed in this lab and chosen among four clip rules with these simulations. Its rule has a reason that does not depend on the model (findings 2.1.0, section 10.3), but other simulators and real ballot data would be the stronger test.
- **Comparison across versions.** `compare.py --force-versions` can pool the collections of two versions; the pooled top layer of 1.0.0 and 2.1.0 differs from 2.1.0's only by Copeland and Score runoff. Its command could join [reference-runs.md](reference-runs.md) as an optional step, with a note that pooled means mix two models.
