"""Pareto-frontier helpers shared by report/generate_figures.py and
presentation/make_figures.py -- both plot fp32/int8 accuracy vs. size/latency/MACs
figures from the same results/ CSVs and used to reimplement this independently."""

import numpy as np


def pareto_frontier(xs, ys):
    """Skyline: sorted by x ascending, keep points whose y beats every prior kept point."""
    pts = sorted(zip(xs, ys), key=lambda p: p[0])
    frontier, best_y = [], -1.0
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def pareto_front_mask(xs, ys):
    """True for points not dominated by any other (lower-or-equal x, higher-or-equal y,
    at least one strict)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dominated = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i != j and xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break
    return ~dominated
