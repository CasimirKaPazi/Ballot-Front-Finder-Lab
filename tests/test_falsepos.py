"""Tests of the false-positive measure (strategy.falsepos). Run from the lab's folder:
python tests/test_falsepos.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods import METHODS  # noqa: E402


def profile(groups):
    """Utilities from (count, utilities) pairs."""
    return np.array([u for k, u in groups for _ in range(k)], float)


# False positives (strategy.falsepos): sincere ballots: nobody changed; under Plurality
# (4 A>B>C, 3 B>C>A, 2 C>B>A) the C voters compromising for B elect B: a gain in the true count, no
# false positive; an A voter deserting to C only loses
from strategy.falsepos import false_positives as _fpos
_Mf = METHODS["Plurality"]
_uf = profile([(4, [1, .5, 0]), (3, [0, 1, .5]), (2, [0, .5, 1])])
_Sf = _Mf.sincere(_uf)
assert _fpos(_Mf, _Sf, _Sf, _uf, np.random.default_rng(0))["changed"] == 0
_Bf = _Sf.copy()
_Bf[-2:] = _Sf[4]
_rf = _fpos(_Mf, _Bf, _Sf, _uf, np.random.default_rng(0), poll_error=1)
assert _rf["coalitions"] == 1 and _rf["fp"] == 0 and _rf["exact_help"] == 1 and np.isclose(_rf["exact_gain"], 0.5), _rf
_Bf = _Sf.copy()
_Bf[0] = _Sf[-1]
_rf = _fpos(_Mf, _Bf, _Sf, _uf, np.random.default_rng(0), poll_error=1)
assert _rf["fp"] == 1 and _rf["fp_hurt"] == 1 and _rf["exact_hurt"] == 1, _rf
assert _rf["patterns"] == {"3>1=2": (1, 1, _rf["gain"])}, _rf["patterns"]   # their own last first
from strategy.falsepos import weak_order as _wo
assert _wo([3, 1, 3], "123") == "1=3>2" and _wo([0, 2, 1], "123") == "2>3>1" and _wo([1], "1") == "1"
print("false positives ok")
