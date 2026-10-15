"""IDS (iterated dropped score), an add-on method.

A score elimination like INS, but instead of stretching every ballot over the remaining candidates,
each round every ballot's lowest-rated remaining candidates are struck out (count 0; a ballot rating
them all the same counts for none) and the other scores stay as given. The lowest total is eliminated
(ties for last split the win), until one is left. A ballot only changes where its own lowest
remaining rating is, and it can be counted by hand.
"""
import numpy as np
import methods
from methods import Method, masks


def dscore_sub(B):
    """For every subset S of candidates: the voter's scores among S with the lowest-rated candidates in
    S set to 0, the others unchanged; 0 outside S. Shape per voter: (2^C, C)."""
    n, C = B.shape
    in_set = masks(C)[None]                    # (1, 2^C, C): True where a candidate is in the set
    scores = B[:, None, :]                     # (n, 1, C): each voter's scores, the same for every set
    lowest = np.where(in_set, scores, np.inf).min(axis=2, keepdims=True)   # (n, 2^C, 1)
    keep = in_set & (scores > lowest)          # in the set and above the set's lowest score
    return np.where(keep, scores, 0.0).reshape(n, -1)


def decide(s):
    """The elimination of INS (methods.d_iterated_normalized_score) on the dropped scores."""
    return methods.d_iterated_normalized_score(s, "dscore_sub")


FEATURES = {"dscore_sub": (dscore_sub, lambda C: (2 ** C, C))}
METHODS = {"IDS": Method("score", decide, ("dscore_sub",))}
GROUP = "rated"


def test():
    """2 voters A5 B4 C0, 1 voter A5 B5 C1, 3 voters A2 B3 C3. Strike each ballot's lowest:
    A5 B4 | A5 B5 | B3 C3 -> A15 B22 C9, C out; over {A,B}: A5 | (tied: none) | B3 -> A10 B9 -> A.
    INS stretches instead (A5 B4 C0 | A5 B5 C0 | A0 B5 C5) and elects B."""
    B = np.array([[5, 4, 0]] * 2 + [[5, 5, 1]] + [[2, 3, 3]] * 3, dtype=float)
    assert np.allclose(METHODS["IDS"].tally(B), [1, 0, 0])
    assert np.allclose(methods.METHODS["INS"].tally(B), [0, 1, 0])
