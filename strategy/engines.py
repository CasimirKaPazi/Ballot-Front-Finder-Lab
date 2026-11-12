"""The strategy engines and their run parameters (run.py --engine; docs/model.md, Strategy engines).

An engine answers the poll rounds (strategy.rounds runs them): it turns the polls of a round into the
voters' ballots. The engines share the polls, the voters who take part (always strategic, adaptive,
sincere) and the rounds; they differ in how a voter finds and judges a move. Some run parameters
belong to some engines only (ENGINE_PARAMS); a results file stores them blank for the others."""


# The strategy engines (run.py --engine), the default first.
ENGINES = {
    "pooled": "individual voters with pooled discovery: moves found and valued per type of voter "
              "(their order of the poll's top three) by counting the round's polls again (pooled.answer_pooled)",
    "arithmetic": "individual voters reading the average of the polls, moves valued by their worth in "
                  "switching voters (individual.answer, individual._value)",
    "exact": "individual voters as arithmetic, moves valued by counting the poll again (individual._value_exact)",
    "groups": "ranking groups: strategic voters with the same sincere ranking answer as a bloc (the engine "
              "of version 1.0.0)",
    "inertia": "experimental (docs/strategy-design.md): voters reason from their honest ballot; a move of their "
               "side in a race among the poll's top three, valued for an unknown share of the side and remembered "
               "over the rounds, is cast if it beats the honest ballot by more than their inertia "
               "(inertia.InertiaEngine)",
}


DEFAULT_ENGINE = "pooled"


# Run parameters that only some engines read: {parameter: (engines, default)}. A run with another
# engine stores them blank (engine_params), so a results file only shows what had an effect.
ENGINE_PARAMS = {
    "aggressive": (("groups",), 0.5),                         # the fixed aggressive/kind split
    "risk": (("groups",), 0),                                 # careful or bold by satisfaction instead
    "risk_tolerance": (("groups",), 0.1),
    "campaign": (("groups", "arithmetic", "exact"), 0.0),     # candidates recommend moves
    "coordination": (("pooled", "arithmetic", "exact", "groups"), 0.25),   # groups: only with campaigns
    "honesty": (("pooled", "arithmetic", "exact"), 0.3),
    "inertia": (("inertia",), 1.0),                           # the largest inertia of an adaptive voter
}


def _number(x, default):
    return float(default if x in (None, "") else x)


def engine_of(p):
    """The engine of a run from its parameters (numbers or text, as in a results file). Results files
    from before --engine (version 3.0.0) have --individual and --response instead: individual 0 (or
    missing) is the ranking groups, individual 1 the engine named by response (arithmetic if missing)."""
    e = p.get("engine")
    if e not in (None, ""):
        return str(e)
    if _number(p.get("individual"), 0) == 0:
        return "groups"
    r = p.get("response")
    return "arithmetic" if r in (None, "") else str(r)


def engine_uses(p):
    """The engine-specific parameters (ENGINE_PARAMS) that have an effect in a run with parameters p:
    those of its engine, except, for the ranking groups, the aggressive share with --risk 1 (risk
    decides who is bold), the risk tolerance with --risk 0, and the coordination share without campaigns."""
    e = engine_of(p)
    used = {k for k, (es, _) in ENGINE_PARAMS.items() if e in es}
    if e == "groups":
        used.discard("risk_tolerance" if _number(p.get("risk"), 0) == 0 else "aggressive")
        if _number(p.get("campaign"), 0) == 0:
            used.discard("coordination")
    return used


def engine_params(p):
    """p (a run's parameters) with "engine" set (engine_of), the older "individual" and "response" removed,
    the engine-specific parameters it uses filled with their default where missing or blank, and the
    others blank ("")."""
    q = {k: v for k, v in p.items() if k not in ("individual", "response")}
    q["engine"] = engine_of(p)
    used = engine_uses(q)
    for k, (_, default) in ENGINE_PARAMS.items():
        q[k] = (default if q.get(k) in (None, "") else q[k]) if k in used else ""
    return q
