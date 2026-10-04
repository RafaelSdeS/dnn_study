"""ml.reporting's test-set statistics: Wilson interval and exact McNemar test."""
import pytest

from ml.reporting import holm, mcnemar_p, wilson_ci


def test_wilson_ci_matches_the_textbook_value():
    lo, hi = wilson_ci(45, 100)  # Wilson 1927 / Agresti & Coull 1998: 45/100 -> [0.3561, 0.5476]
    assert (lo, hi) == pytest.approx((0.3561, 0.5476), abs=1e-4)


def test_mcnemar_counts_only_the_images_one_model_gets_right():
    both = [True] * 50
    assert mcnemar_p(both + [True] * 10, both + [False] * 10) == pytest.approx(2 / 2 ** 10)  # 10 vs 0 discordant
    assert mcnemar_p(both + [True, False], both + [False, True]) == 1.0  # 1 vs 1: no evidence
    assert mcnemar_p(both, both) == 1.0


def test_holm_step_down():
    # sorted p 0.01, 0.02, 0.04 among 3 tests -> 0.03, 0.04, 0.04 (monotone); NaN untouched, not counted
    out = holm([0.04, float("nan"), 0.01, 0.02])
    assert out[0] == pytest.approx(0.04) and out[2] == pytest.approx(0.03) and out[3] == pytest.approx(0.04)
    assert out[1] != out[1]


def test_pareto_front_keeps_only_undominated_points():
    from scripts.phase11.design_figures import pareto_front

    # (cost, acc): (2, 25) loses to (2, 30) at equal cost, (3, 20) to the cheaper (2, 30)
    assert pareto_front([1, 2, 3, 2, 4], [10, 30, 20, 25, 40]).tolist() == [True, True, False, False, True]
