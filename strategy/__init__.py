"""Strategic voting in poll rounds (respond): a single election with a few polls before it.
  Round 0: every voter votes sincerely ("naive", uninformed).
  Round r (1 ... rounds): a poll of round r-1 is published, and every strategic voter answers it
  at the same time; nobody sees anyone else's answer before the next poll.
  An engine answers the polls (run.py --engine, engines.ENGINES): individual voters, each deciding
  alone from the polls (pooled.py: pooled, the default; individual.py: arithmetic, exact), or the
  ranking groups (groups.py: groups, the engine of version 1.0.0). The polls are the same for all.
  - A poll publishes only the method's own counts (the summed ballot features the method decides
    from: first preferences for Plurality, approvals for Approval, the pairwise matrix for
    Condorcet methods, ...), from a random sample of `poll_size` voters (without replacement),
    scaled up to the electorate. `samples` independent polls (5) stand for the uncertainty.
    Nobody sees anyone's preferences or individual ballots. Only totals a real poll could show are
    published (per candidate, per pair, per grade): methods counted from totals over every subset of
    candidates are counted by the voters with a stand-in (Method.poll_view, POLL in methods.py).
  - Uncertainty: voters treat a poll as `poll_error` times less precise than its sampling error
    (default 2: real polls miss results by about twice their stated sampling error, Shirani-Mehr,
    Rothschild, Goel & Gelman 2018). The polls are spread around the true count by that factor.
  - Settled: round `rounds` gives the same ballots as the round before (a fixed point).
  rounds = 0 gives the sincere ballots.

Partial strategy: `strategic` marks the voters who take part (a random share of the
electorate, drawn in run.py, the same voters for every method). The others always vote sincerely.
Adaptive voters (`adaptive`: one personal threshold per voter, drawn once in run.py, uniform on 0..1,
nan = not adaptive) decide each round whether to vote strategically, as their engine has it: the
ranking groups by their stake in the race between the two leaders (groups.py), individual voters by
whether a move gains more than their cost of voting insincerely (their threshold x --honesty). The
shares strategic per round go into `activated`.

The modules:
  engines.py      the engines and their run parameters (ENGINES, ENGINE_PARAMS, engine_params)
  rounds.py       the poll rounds every engine runs in (respond)
  polls.py        polls, their deviations and leaders
  measures.py     sincerity, fidelity, implied values, kinds of strategy
  groups.py       the ranking groups (--engine groups)
  individual.py   individual voters reading the average of the polls (--engine arithmetic, exact)
  pooled.py       pooled discovery (--engine pooled)
  inertia.py      reasoning from home, sides and inertia (--engine inertia, experimental)
  common.py       what the engines of individual voters share (cost, trust, discovery stream)
  moves.py        moves as steps, and the ballots that follow from them (individual voters)
"""
from .engines import ENGINES, DEFAULT_ENGINE, ENGINE_PARAMS, engine_of, engine_uses, engine_params  # noqa: F401
from .measures import sincerity, fidelity, implied_values, strategy_types, worst_reversal  # noqa: F401
from .polls import default_poll_size, poll_tallies  # noqa: F401
from .rounds import respond  # noqa: F401
