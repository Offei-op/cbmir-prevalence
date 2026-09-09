import numpy as np
import pytest
from cbmir.retrieval import query_metrics
from cbmir.analysis import paired_summary, holm


def test_sparse_relevance_ndcg_ideal_is_not_full_k():
    relevance = np.array([1, 0, 0, 0])
    p, n = query_metrics(np.array([[0, 1, 2], [1, 0, 2]]), relevance, 3)
    np.testing.assert_allclose(p, [1 / 3, 1 / 3])
    np.testing.assert_allclose(n, [1, 1 / np.log2(3)])


def test_no_relevant_items():
    p, n = query_metrics(np.array([[0, 1]]), np.zeros(2), 2)
    assert p[0] == n[0] == 0


def test_reject_invalid_rank():
    with pytest.raises(ValueError):
        query_metrics(np.array([[2]]), np.array([1]), 1)


def test_identical_models_have_valid_comparison():
    r = paired_summary([0, 0.5, 1], [0, 0.5, 1], samples=30)
    assert r["p_raw"] == 1 and r["ci_low"] == r["ci_high"] == 0


def test_holm_preserves_original_order():
    np.testing.assert_allclose(holm([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])


def test_positive_paired_effect():
    r = paired_summary([0.5] * 10, [0.2] * 10, samples=30)
    assert r["mean_difference"] == pytest.approx(0.3)
    assert r["ci_low"] == pytest.approx(0.3)
