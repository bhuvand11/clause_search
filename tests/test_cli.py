import pytest

from src.cli import find_contract
from src.data import contract_titles

TITLES = {
    "cuad/CANO.txt": "Sponsorship Agreement between R. C. Boyd Enterprises, LLC and Cano Petroleum, Inc.",
    "cuad/ARC.txt": "Sponsorship Agreement between Jacksonville Jaguars, LLC and The ARC Group, Inc.",
}


def test_find_contract_by_unique_substring():
    assert find_contract("cano petroleum", TITLES) == "cuad/CANO.txt"


def test_find_contract_rejects_ambiguous_or_unknown_names():
    with pytest.raises(SystemExit):
        find_contract("Sponsorship", TITLES)
    with pytest.raises(SystemExit):
        find_contract("Netflix", TITLES)


def test_contract_titles_come_from_query_prefix():
    queries = [{"query": "Consider the Gas Franchise Agreement; What is the term?",
                "snippets": [{"file_path": "cuad/GAS.txt"}]}]
    assert contract_titles(queries) == {"cuad/GAS.txt": "Gas Franchise Agreement"}
