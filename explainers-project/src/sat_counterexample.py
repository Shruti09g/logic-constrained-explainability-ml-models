# src/sat_counterexample.py
"""
SAT-guided counterexample search (CEGAR-style) over a discretized formal region.

Approach:
1. Discretize each numeric interval into `num_bins` bins; represent each bin by a propositional variable.
   For categorical features, create one variable per allowed category value.
2. Build CNF constraints that ensure exactly-one option is chosen per feature.
3. Query a SAT solver (PySAT) for a satisfying assignment (a full discrete sample inside the region).
4. Decode the assignment into a concrete sample by taking bin midpoints for numeric features and chosen
   categories for categorical ones.
5. Evaluate the model prediction and the heuristic assertion (LIME/SHAP/etc) on the decoded sample.
   - If model prediction == original_prediction AND heuristic_assertion(sample) == False:
       → we found a counterexample (return it).
   - Else: block this exact discrete assignment (add a clause that forbids this assignment) and continue.
6. Repeat until solver returns UNSAT or until max_tries is reached.

Requirements:
  pip install python-sat
(If python-sat is not installed, the module raises ImportError.)

Limitations:
- This is a discretized search; if counterexamples exist only at values between bin midpoints, you may miss them.
- Increasing num_bins increases precision but also exponentially increases SAT search space.
- This is a pragmatic, sound approach: any returned counterexample is real (checked by the model+heuristic).
"""

from typing import Dict, Any, List, Tuple, Callable, Optional
import math
import numpy as np
import pandas as pd

# PySAT imports
try:
    from pysat.formula import CNF
    from pysat.solvers import Solver
except Exception as e:
    raise ImportError("python-sat (PySAT) is required. Install with `pip install python-sat`") from e


def _make_bin_midpoints(lo: float, hi: float, num_bins: int) -> List[float]:
    """Return list of bin midpoints for interval [lo,hi]."""
    if lo >= hi:
        return [(lo + hi) / 2.0]
    edges = np.linspace(lo, hi, num_bins + 1)
    mids = []
    for i in range(num_bins):
        mids.append(float((edges[i] + edges[i + 1]) / 2.0))
    return mids


def _add_exactly_one(cnf: CNF, lits: List[int]) -> None:
    """
    Add at-least-one and at-most-one clauses for integer literals (PySAT uses integers).
    at-least-one: (l1 v l2 v ... v ln)
    at-most-one: pairwise (~li v ~lj) for all i < j
    """
    if not lits:
        return
    cnf.append(lits[:])  # at least one
    # pairwise at-most-one (naive)
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            cnf.append([-lits[i], -lits[j]])


class SATDiscretizer:
    """
    Manage mapping between features and SAT variables, and decode/encode assignments.
    """

    def __init__(self):
        # mapping: (feature, option_index) -> var (int)
        self.varmap: Dict[Tuple[str, int], int] = {}
        self.revmap: Dict[int, Tuple[str, int]] = {}
        self.counter = 1

        # per-feature option metadata for decoding
        self.options_meta: Dict[str, List[Any]] = {}

    def new_option(self, feature: str, meta: Any) -> int:
        """
        Register a new option for feature, return sat variable id.
        meta: any data required to decode (e.g., category string or numeric midpoint)
        """
        idx = len(self.options_meta.get(feature, []))
        var = self.counter
        self.counter += 1
        self.varmap[(feature, idx)] = var
        self.revmap[var] = (feature, idx)
        self.options_meta.setdefault(feature, []).append(meta)
        return var

    def get_options(self, feature: str) -> List[Any]:
        return self.options_meta.get(feature, [])

    def decode_assignment(self, model: List[int]) -> Dict[str, Any]:
        """
        Given a SAT model (list of ints: positive means true), decode to concrete sample.
        """
        assignment = {}
        model_set = set(model)
        for var, (feat, idx) in self.revmap.items():
            if var in model_set:
                # choose this option
                meta = self.options_meta[feat][idx]
                assignment[feat] = meta
        return assignment

    def block_assignment_clause(self, assignment: Dict[str, Any]) -> List[int]:
        """
        Build a blocking clause that forbids the exact discrete assignment.
        Clause is the disjunction of negated chosen literals.
        """
        lits = []
        # find the var for chosen option per feature and add its negation
        for feat, opts in self.options_meta.items():
            chosen_idx = None
            # meta equality to identify which option chosen
            for idx, meta in enumerate(opts):
                if feat in assignment and assignment[feat] == meta:
                    chosen_idx = idx
                    break
            if chosen_idx is not None:
                var = self.varmap[(feat, chosen_idx)]
                lits.append(-var)
        return lits


def build_region_cnf(region: Dict[str, Any], num_bins: int = 8) -> Tuple[CNF, SATDiscretizer]:
    """
    Build CNF encoding of the discretized region.
    region: raw_feature -> ["eq", value] OR [lo, hi]
    For categorical features, region[...] = ["eq", "Category"] or list of allowed categories.
    Returns (CNF, SATDiscretizer).
    """
    cnf = CNF()
    disc = SATDiscretizer()

    for feat, cons in region.items():
        # equality form
        if isinstance(cons, list) and len(cons) == 2 and cons[0] == "eq":
            val = cons[1]
            # if value is None, we treat it as unconstrained: add bins across reasonable range (not supported here)
            # For equality, just create a single option equal to that value
            var = disc.new_option(feat, val)
            _add_exactly_one(cnf, [var])
        else:
            # numeric interval case: [lo, hi]
            try:
                lo = float(cons[0])
                hi = float(cons[1])
                # choose bin midpoints
                mids = _make_bin_midpoints(lo, hi, num_bins)
                vars_for_feat = []
                for m in mids:
                    v = disc.new_option(feat, m)
                    vars_for_feat.append(v)
                _add_exactly_one(cnf, vars_for_feat)
            except Exception:
                # fallback: treat as categorical or single equality
                if isinstance(cons, list):
                    # treat each element as one allowed category
                    vars_for_feat = []
                    for cat in cons:
                        v = disc.new_option(feat, cat)
                        vars_for_feat.append(v)
                    _add_exactly_one(cnf, vars_for_feat)
                else:
                    v = disc.new_option(feat, cons)
                    _add_exactly_one(cnf, [v])

    return cnf, disc


def find_counterexample_with_sat(
    region: Dict[str, Any],
    original_instance: pd.Series,
    model_predict_fn: Callable[[pd.DataFrame], Any],
    heuristic_assertion_fn: Callable[[pd.DataFrame], bool],
    num_bins: int = 8,
    max_tries: int = 1000,
    solver_name: str = "glucose3",
    verbose: bool = False
) -> Optional[pd.Series]:
    """
    CEGAR loop:
    - Build CNF for discretized region
    - While solver says SAT:
        - decode candidate
        - build concrete sample DataFrame
        - evaluate model_predict and heuristic_assertion
        - if (model predicts same label) and (heuristic_assertion == False) -> return sample (real counterexample)
        - else add blocking clause to CNF to forbid this assignment and continue
    Returns the concrete sample (pandas Series) if found, otherwise None.
    """
    cnf, disc = build_region_cnf(region, num_bins=num_bins)

    # use solver
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
        tries = 0
        while tries < max_tries and solver.solve():
            tries += 1
            model = solver.get_model()  # list of ints (positive = true)
            # decode assignment -> mapping feat->value (midpoints or categories)
            assignment = disc.decode_assignment(model)
            # build DataFrame row from original_instance with replaced values for features present
            candidate = original_instance.copy()
            for f, v in assignment.items():
                # if original is int, convert to int when v is near-int
                try:
                    if pd.isna(candidate.get(f)) and isinstance(v, (int, float)):
                        candidate[f] = v
                    else:
                        candidate[f] = type(candidate[f])(v) if candidate.get(f) is not None and not pd.isna(candidate.get(f)) and not isinstance(candidate.get(f), str) else v
                except Exception:
                    candidate[f] = v

            # create single-row DataFrame (raw)
            candidate_df = pd.DataFrame([candidate])

            # evaluate model prediction
            try:
                pred = model_predict_fn(candidate_df)
                # assume model_predict_fn returns array-like of labels
                label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            except Exception as e:
                # if model fails on this candidate, block and continue
                if verbose:
                    print(f"[SAT] model failed on candidate {assignment}: {e}")
                # block this assignment
                block_clause = disc.block_assignment_clause(assignment)
                if block_clause:
                    solver.add_clause(block_clause)
                continue

            # check original prediction to ensure we keep same label
            try:
                orig_pred = model_predict_fn(pd.DataFrame([original_instance]))  # original
                orig_label = int(orig_pred[0]) if hasattr(orig_pred, "__len__") else int(orig_pred)
            except Exception as e:
                raise RuntimeError(f"Original model prediction failed: {e}")

            if label != orig_label:
                # candidate does not respect fixed model prediction; block and continue
                if verbose:
                    print(f"[SAT] candidate changed prediction: {label} != {orig_label}, blocking.")
                block_clause = disc.block_assignment_clause(assignment)
                if block_clause:
                    solver.add_clause(block_clause)
                continue

            # evaluate heuristic: we want heuristic_assertion_fn(candidate_row) == False to be a counterexample
            try:
                holds = heuristic_assertion_fn(candidate_df.iloc[0])
            except Exception as e:
                # if heuristic fails to evaluate, treat as not holding (counterexample) ??? safer: block and continue
                if verbose:
                    print(f"[SAT] heuristic evaluation raised: {e}. Blocking candidate.")
                block_clause = disc.block_assignment_clause(assignment)
                if block_clause:
                    solver.add_clause(block_clause)
                continue

            if not holds:
                # FOUND real counterexample
                if verbose:
                    print(f"[SAT] Found counterexample after {tries} tries: {assignment}")
                # return as pandas Series
                return candidate

            # else heuristic holds: block this exact discrete assignment and continue
            block_clause = disc.block_assignment_clause(assignment)
            if block_clause:
                solver.add_clause(block_clause)

        # if loop exits, no candidate found
        if verbose:
            print("[SAT] No counterexample found (UNSAT or max tries reached).")
        return None
