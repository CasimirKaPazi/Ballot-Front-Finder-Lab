"""The poll rounds every engine runs in (respond; the package docstring). The rounds, the polls and the
voters who take part are the same for every engine; each round the engine (engines.ENGINES) turns the
round's polls into the voters' ballots."""
import numpy as np

from .engines import ENGINES
from .measures import sincerity
from .polls import default_poll_size


class Context:
    """A run's state shared by the rounds and the engine. The engine reads it and sets B (through
    respond: the ballots it returns) and active (the voters strategic this round).
      method, u, rng, n, C      the method, the utilities (voters x candidates), the run's random stream
      sincere, B                the sincere ballots; the ballots of the last round (the first poll's)
      fixed, adapt, q_adapt     always strategic voters; adaptive voters; their personal thresholds (inf
                                for the others); n_adaptive their number
      active                    the voters strategic this round (at first the always strategic ones)
      pm, inc                   the method as voters count a poll (Method.poll_view); whether it has feature
                                sums (polls then counted from totals)
      samples, m, m_full        polls per round; voters per poll; per poll without feature sums
      poll_error, search        --poll-error; --search
      leaders                   a dict to fill with {round: the two poll leaders}, or None"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _engine(name):
    from .groups import GroupsEngine
    from .individual import ArithmeticEngine, ExactEngine
    from .pooled import PooledEngine
    from .inertia import InertiaEngine
    return {"groups": GroupsEngine, "arithmetic": ArithmeticEngine, "exact": ExactEngine,
            "pooled": PooledEngine, "inertia": InertiaEngine}[name]


def respond(method, u, rng, rounds=1, samples=10, poll_size=0, search=1, strategic=None, poll_error=2.0,
            report=None, aggressive=None, start=None, adaptive=None, activated=None, campaign=None, risk=None,
            cstats=None, leaders=None, engine="groups", coordination=0.25, honesty=0.3, inertia=1.0,
            inertia_shape=1.0, shares=4, evidence=4.0, memory=0.8, race=None):
    """Poll rounds (see the package docstring). Returns ballots of the last round, whether it
    settled (same as the round before), and mean sincerity (all voters).
    report: a list of round numbers; then `rounds` is ignored, the rounds run up to the highest,
    and the result is a list of such triples, one per listed round. Round r here is exactly what
    rounds = r gives (each round depends only on the ones before it, and draws its random numbers
    in sequence), so one run serves a sweep over the number of rounds.
    start: ballots (n x C) the first poll is drawn from instead of the sincere ones (run.py --start
    previous: a previous election's result); only the information of round 1 changes.
    adaptive: personal thresholds of the adaptive voters (one per voter, nan = not adaptive; None =
    none); how they decide each round is the engine's (groups: by their stake in the race between the
    two leaders; individual voters: by their cost of voting insincerely). activated: a dict, filled with
    {round: share of the adaptive voters strategic in that round} (0 for round 0).
    cstats: a dict, filled with {round: {"follow": share of all voters following a campaign
    recommendation, "top": candidate whose recommendation has the most followers (-1: none), "bold":
    share of the strategic voters who are bold (--risk; else nan), "changed": share of all voters whose
    ballot differs from the round before}}. leaders: a dict, filled with
    {round: the round's two poll leaders (a, b)}, as the polls the voters answered showed them.
    engine: which engine answers the polls (engines.ENGINES), with its options (each engine reads its
    own and ignores the others'):
      groups      aggressive (per voter, aggressive or kind; None = all aggressive), campaign, risk
                  (None: off, the fixed aggressive split; or {"threshold": one per voter, 0..1, "kappa":
                  tolerance factor}: careful or bold by their satisfaction with the polls)
      arithmetic, exact   coordination, honesty, campaign
      pooled      coordination, honesty
      inertia     inertia (the largest inertia of an adaptive voter, a share of their utility range),
                  inertia_shape, shares, evidence, memory (inertia.InertiaEngine)
    campaign: None (off) or {"share": phi, "cand_u": candidates' utilities, C x C}: every candidate
    recommends a move each round (docs/campaigns.md). coordination: the share of their camp (pooled: of
    their type) an individual voter expects to vote as they do (0: only themselves), or per voter a
    number of voters. honesty: an adaptive individual voter's cost of voting insincerely is their
    personal draw times this, a share of their utility range.
    race: None, or a function race(r, B, polls) called after every round r from 1 but the last (strategic
    nomination in the race, nomination.Race): B the round's ballots, polls the round's poll samples
    (voter indices; without strategic voters drawn for it). It returns None (the field stays)
    or (u, B) for a new field: the voters' utilities there and the ballots carried to it. The rounds
    then go on there with a fresh engine (memory and trust start again); snapshots and sincerity
    are on the field of their round."""
    options = {"aggressive": aggressive, "campaign": campaign, "risk": risk, "coordination": coordination,
               "honesty": honesty, "inertia": inertia, "inertia_shape": inertia_shape, "shares": shares,
               "evidence": evidence, "memory": memory}
    if report is not None:
        want = sorted(set(int(r) for r in report))
        if want[0] < 0:
            raise ValueError("poll rounds must be 0 or more")
        snaps = {}
        _respond(method, u, rng, max(want), samples, poll_size, search, strategic, poll_error,
                 lambda r, res: snaps.__setitem__(r, res) if r in want else None, start, adaptive, activated,
                 cstats, leaders, engine, options, race)
        return [snaps[int(r)] for r in report]
    return _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, None, start,
                    adaptive, activated, cstats, leaders, engine, options, race)


def _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, snapshot, start,
             adaptive, activated, cstats, leaders, engine, options, race=None):
    if engine not in ENGINES:
        raise ValueError(f"unknown strategy engine {engine!r} (one of {', '.join(ENGINES)})")
    n, C = u.shape
    fixed = np.ones(n, bool) if strategic is None else np.asarray(strategic, bool).copy()
    q = None if adaptive is None else np.asarray(adaptive, float)
    adapt = np.zeros(n, bool) if q is None else ~np.isnan(q) & ~fixed
    q_adapt = np.where(adapt, np.nan_to_num(q, nan=np.inf), np.inf) if q is not None else np.full(n, np.inf)
    pm = method.poll_view()                    # how voters count a poll (docs/model.md, What a poll shows)
    m = min(n, poll_size if poll_size > 0 else default_poll_size(n))
    # methods without feature sums can't scale counts: a smaller poll has about poll_error times the error
    # (only approximates poll_error >= 1; smaller values give polls of poll_size voters)
    m_full = max(1, min(m, int(round(m / poll_error ** 2)))) if poll_error > 0 else m
    cls = _engine(engine)

    def build(u, B):
        """The context and a fresh engine on a field (voters' utilities u), from ballots B (None: sincere)."""
        sincere = method.sincere(u)
        ctx = Context(method=method, u=u, rng=rng, n=n, C=u.shape[1], sincere=sincere,
                      B=sincere.copy() if B is None else np.array(B, dtype=float), fixed=fixed,
                      adapt=adapt, q_adapt=q_adapt, n_adaptive=int(adapt.sum()), active=fixed.copy(), pm=pm,
                      inc=pm.full is None, samples=samples, m=m, m_full=m_full, poll_error=poll_error,
                      search=search, leaders=leaders)
        return ctx, cls(ctx, **{k: v for k, v in options.items() if k in cls.OPTIONS})
    ctx, eng = build(u, None)

    def record(r, follow=0.0, top=-1, changed=0.0):
        if activated is not None:
            activated[r] = float((ctx.active & adapt).sum()) / ctx.n_adaptive if ctx.n_adaptive else 0.0
        if cstats is not None:
            cstats[r] = {"follow": follow, "top": top, "bold": eng.bold(), "changed": changed}

    def draw():
        return np.stack([rng.choice(n, m if ctx.inc else m_full, replace=False) for _ in range(samples)])

    def nominate(r, polls):
        """Strategic nomination after round r (race): True if the field changed (ctx and engine are new)."""
        nonlocal ctx, eng
        if race is None or r >= rounds:
            return False
        new = race(r, ctx.B, polls)
        if new is None:
            return False
        ctx, eng = build(*new)
        return True
    record(0)
    if snapshot:
        snapshot(0, (ctx.B, True, 1.0))
    if rounds <= 0:
        return ctx.B, True, 1.0
    if not fixed.any() and not adapt.any():    # nobody votes strategically: sincere in every round
        for r in range(1, rounds + 1):
            record(r)
            if snapshot:
                snapshot(r, (ctx.B, True, 1.0))
            if race is not None and nominate(r, draw()):
                ctx.B = ctx.sincere.copy()          # sincere on the new field too
        return ctx.B, True, 1.0
    if start is not None:                      # the first poll shows these ballots (a previous result)
        ctx.B = np.array(start, dtype=float)
    settled = False
    for r in range(1, rounds + 1):
        polls = draw()
        B_prev = ctx.B
        ctx.B, follow, top = eng.round(r, polls)
        diff = np.any(ctx.B != B_prev, axis=1)
        record(r, follow, top, float(diff.mean()))
        settled = not diff.any()                   # the ballots repeat the round before
        if snapshot:
            snapshot(r, (ctx.B, settled, float(sincerity(method, ctx.B, ctx.u).mean())))
        if nominate(r, polls):
            settled = False
    return ctx.B, settled, float(sincerity(method, ctx.B, ctx.u).mean())
