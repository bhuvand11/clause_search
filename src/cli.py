"""Command-line clause search.

Examples:
    python -m src.cli "Consider the Gas Franchise Agreement between the Town of Vinton and Roanoke Gas Company; What is the expiration date?"
    python -m src.cli "Can this agreement be assigned?" --contract "Cano Petroleum"
    python -m src.cli --list-contracts
    python -m src.cli "What is the governing law?" --contract Veoneer --json
"""

import argparse
import json
import sys
import textwrap

from src.data import contract_titles, load_benchmark
from src.pipeline import Pipeline, load_config


def find_contract(name: str, titles: dict[str, str]) -> str:
    """Resolve a case-insensitive substring of a contract's title or file name."""
    matches = [fp for fp, t in titles.items() if name.lower() in (t + " " + fp).lower()]
    if len(matches) != 1:
        options = "\n  ".join(titles[fp] for fp in matches) if matches else "(none)"
        sys.exit(f"'{name}' matches {len(matches)} contracts; be more specific:\n  {options}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--contract", help="search only this contract (part of its name)")
    parser.add_argument("-k", type=int, help="number of passages (default: config top_k)")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--list-contracts", action="store_true")
    args = parser.parse_args()

    queries, corpus = load_benchmark()
    titles = contract_titles(queries)
    if args.list_contracts:
        for fp in sorted(titles, key=titles.get):
            print(titles[fp])
        return
    if not args.question:
        parser.error("a question is required")

    contract = find_contract(args.contract, titles) if args.contract else None
    pipeline = Pipeline(load_config(args.config), corpus)
    results, reason = pipeline.search(args.question, args.k, contract)

    if args.json:
        print(json.dumps({"question": args.question, "abstained": reason, "results": results}, indent=2))
        return
    if reason:
        print(f"No relevant clause found: {reason}.")
        return
    for rank, r in enumerate(results, 1):
        print(f"\n[{rank}] {titles.get(r['file_path'], r['file_path'])}")
        print(f"    {r['file_path']}  chars {r['start']}-{r['end']}  score {r['score']:.2f}")
        body = " ".join(r["text"].split())
        print(textwrap.indent(textwrap.fill(body, 96), "    "))


if __name__ == "__main__":
    main()
