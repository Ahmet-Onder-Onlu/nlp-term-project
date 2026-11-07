from chronofuse.eval.metrics import per_query_ndcg
from chronofuse.indexing.base import maxp
from chronofuse.retrieval.fusion import rrf


def test_rrf_prefers_consensus():
    fused = rrf({"a": [("d1", 9), ("d2", 8)], "b": [("d2", 5), ("d3", 4)]})
    assert fused[0][0] == "d2"


def test_maxp_aggregates_passages():
    r = maxp(["x#0", "y#0", "x#1"], [1.0, 2.0, 3.0], k=10)
    assert r == [("x", 3.0), ("y", 2.0)]


def test_missing_query_scores_zero():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    s = per_query_ndcg({"q1": [("d1", 1.0)]}, qrels)
    assert s["q1"] == 1.0 and s["q2"] == 0.0
