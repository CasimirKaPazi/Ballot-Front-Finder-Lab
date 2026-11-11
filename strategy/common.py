"""What the engines of individual voters (arithmetic, exact, pooled) share: each voter's cost of voting
insincerely, their trust in moves (judge, trust, aims: a move that backfired loses trust in what it
aims at) and the random stream of discovery."""
import numpy as np


class IndividualVoters:
    """The state every engine of individual voters keeps across the rounds. ctx: the run's shared state
    (rounds.Context); coordination: the share of their camp (pooled: of their type) a voter expects to
    vote as they do, or per voter a number of voters; honesty: an adaptive voter's cost of voting
    insincerely is their personal draw (ctx.q_adapt) times this, a share of their utility range."""
    OPTIONS = ("coordination", "honesty")

    def __init__(self, ctx, coordination=0.25, honesty=0.3):
        self.ctx = ctx
        # each voter's cost of voting insincerely (a share of their utility range): always strategic 0,
        # adaptive their personal draw x --honesty, sincere never
        self.threshold = np.where(ctx.fixed, 0.0, np.where(ctx.adapt, ctx.q_adapt * honesty, np.inf))
        self.coordination = coordination
        self.memory = {"last": {}, "trust": {}}   # what each voter's move promised, and their trust in moves
        # discovery draws from its own stream (spawned, so the polls stay those of the run without it)
        self.rng = np.random.default_rng(ctx.rng.bit_generator.seed_seq.spawn(1)[0])

    def bold(self):
        return float("nan")


RATIO_LOW = -3.0     # judge: a move can backfire up to three times as badly as it promised


def aims(move):
    """What a move aims at: ("up", c) for each candidate it raises, ("down", c) for each it lowers, and
    ("campaign", c) when it follows candidate c's recommendation. Trust attaches to these, not to the exact ballot, so that a distrusted aim is not simply tried again in
    another form."""
    out = set()
    for st in move:
        if st[0] in ("top", "upto", "above"):
            out.add(("up", st[1]))
        elif st[0] in ("bottom", "downfrom", "below"):
            out.add(("down", st[1]))
        elif st[0] == "campaign":                 # following candidate c's campaign: trust in c's campaigns
            out.add(("campaign", st[1]))
    return tuple(sorted(out))


def judge(u, reading, memory):
    """Feedback: each voter who made a move in the last round compares what it promised (the gain in
    their expected outcome over keeping their ballot) with what the new poll shows (their expected
    outcome now against before). The ratio, from RATIO_LOW (backfired three times as badly as
    promised) to 1 (kept its promise), updates their trust in each aim of the move like a running
    average with a prior of one kept promise: trust = (trust * n + ratio) / (n + 1). Trust below 0
    turns the promise into a threat: the voter expects the move to hurt."""
    eu_now = u @ reading.P
    for v, (key, promised, before) in list(memory["last"].items()):
        if promised > 0:
            ratio = float(np.clip((eu_now[v] - before) / promised, RATIO_LOW, 1.0))
            for a in aims(key):
                rho, k = memory["trust"].get((v, a), (1.0, 1))
                memory["trust"][(v, a)] = ((rho * k + ratio) / (k + 1), k + 1)
    memory["last"] = {}


def trust(memory, v, move):
    """A voter's trust in a move: the least trust in any of its aims (1 if never tried)."""
    return min([memory["trust"].get((v, a), (1.0, 1))[0] for a in aims(move)] or [1.0])


class _PollChances:
    """What judge needs of a reading: the winner chances of the round's polls."""
    def __init__(self, P):
        self.P = P
