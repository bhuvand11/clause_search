from src.pipeline import Pipeline, parse_query
from src.retrieval import DocRouter

CORPUS = {
    "cuad/AcmeCorp_Sponsorship.txt": "Sponsorship Agreement between Acme Corp and Boyd LLC. "
                                     "This Agreement is governed by the laws of Texas. " * 3,
    "cuad/ZenithInc_Supply.txt": "Supply Agreement between Zenith Inc and Orbit Ltd. "
                                 "Neither party may assign this Agreement without consent. " * 3,
}
CONFIG = {
    "chunking": {"method": "sentence", "size": 120},
    "retrieval": {"method": "bm25"},
    "query": {"strip_title": True},
    "routing": {"enabled": True, "top_docs": 1},
    "abstention": {"min_doc_score": 0.5},
    "top_k": 2,
}


def test_parse_query_splits_title_and_question():
    assert parse_query("Consider the Supply Agreement between A and B; What is the governing law?") == (
        "Supply Agreement between A and B", "What is the governing law?")
    assert parse_query("What is the capital of Japan?") == (None, "What is the capital of Japan?")


def test_router_picks_named_contract_and_scores_unknown_low():
    router = DocRouter(CORPUS)
    assert router.rank("Supply Agreement between Zenith Inc and Orbit Ltd")[0] == ("cuad/ZenithInc_Supply.txt", 1.0)
    assert router.rank("License between Aurora Pictures and Bluewave")[0][1] < 0.5


def test_search_returns_passages_from_routed_contract_with_offsets():
    p = Pipeline(CONFIG, CORPUS)
    results, reason = p.search("Consider the Acme Corp and Boyd sponsorship; What law governs?")
    assert reason is None
    assert {r["file_path"] for r in results} == {"cuad/AcmeCorp_Sponsorship.txt"}
    for r in results:
        assert CORPUS[r["file_path"]][r["start"]:r["end"]] == r["text"]


def test_search_abstains_for_unknown_contract():
    p = Pipeline(CONFIG, CORPUS)
    results, reason = p.search("Consider the License between Aurora Pictures and Bluewave; What law governs?")
    assert results == [] and "no indexed contract" in reason
