"""Streamlit web app for clause search.

Run with:  streamlit run app.py
"""

import html
import re
import time

import streamlit as st

from src.data import contract_titles, load_benchmark
from src.pipeline import Pipeline, load_config

CONFIG_PATH = "configs/final.yaml"
CONTEXT_CHARS = 180
AUTO = "Detect from question"

# Example questions: (label, contract keyword, question). Two of them show the "no clause" path.
EXAMPLES = [
    ("Assignment", "Cano Petroleum", "Can this agreement be assigned to another company?"),
    ("Expiry", "Roanoke Gas", "When does the franchise expire?"),
    ("Governing law", "Veoneer", "Which country's law governs this agreement?"),
    ("Insurance", "Cano Petroleum", "What insurance must the parties carry?"),
    ("Unknown contract", None, "Consider the Content License Agreement between Aurora Pictures and "
                               "Bluewave Streaming; Can the licensee sublicense the content?"),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput input, .stSelectbox, button { font-family: 'Inter', sans-serif; }
.block-container { max-width: 860px; padding-top: 2.6rem; }
.eyebrow { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #6B6B6B; margin-bottom: .2rem; }
.title { font-family: 'Source Serif 4', Georgia, serif; font-size: 2.3rem; font-weight: 600; color: #1F1F1F; margin: 0; line-height: 1.15; }
.lede { color: #4A4A4A; font-size: .98rem; margin: .45rem 0 1.4rem 0; max-width: 620px; }
.status { color: #5A5A5A; font-size: .85rem; margin: .4rem 0 .9rem 0; }
.card { background: #FFFFFF; border: 1px solid #E4E0D6; border-radius: 6px; padding: 1rem 1.15rem .9rem 1.15rem; margin-bottom: .85rem; }
.card.top { border-left: 3px solid #23395B; }
.meta { display: flex; justify-content: space-between; gap: 1rem; font-size: .78rem; color: #6B6B6B; margin-bottom: .5rem; }
.contract { font-weight: 600; color: #23395B; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rank { font-variant-numeric: tabular-nums; white-space: nowrap; }
.passage { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.0rem; line-height: 1.6; color: #1F1F1F; white-space: pre-wrap; }
.ctx { color: #9A968C; }
.empty { background: #FFFFFF; border: 1px solid #E4E0D6; border-radius: 6px; padding: 1.1rem 1.2rem; }
.empty h4 { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; margin: 0 0 .35rem 0; font-size: 1.15rem; }
.empty p { color: #4A4A4A; margin: 0; font-size: .92rem; }
.empty .detail { color: #8A877F; font-size: .78rem; margin-top: .6rem; }
.side-note { font-size: .8rem; color: #6B6B6B; line-height: 1.5; }
</style>
"""


@st.cache_resource(show_spinner="Loading contracts and models...")
def load():
    queries, corpus = load_benchmark()
    config = load_config(CONFIG_PATH)
    return Pipeline(config, corpus), contract_titles(queries), corpus


def tidy(text: str) -> str:
    """Collapse runs of blank lines and spaces for display (offsets are unaffected)."""
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n\s*\n+", "\n\n", text))


def render_result(rank: int, r: dict, title: str, corpus: dict) -> None:
    doc = corpus[r["file_path"]]
    before = doc[max(0, r["start"] - CONTEXT_CHARS):r["start"]]
    after = doc[r["end"]:r["end"] + CONTEXT_CHARS]
    body = (f'<span class="ctx">{"…" if r["start"] > CONTEXT_CHARS else ""}{html.escape(tidy(before))}</span>'
            f'{html.escape(tidy(r["text"]))}'
            f'<span class="ctx">{html.escape(tidy(after))}{"…" if r["end"] + CONTEXT_CHARS < len(doc) else ""}</span>')
    st.markdown(
        f'<div class="card{" top" if rank == 1 else ""}">'
        f'<div class="meta"><span class="contract">{html.escape(title)}</span>'
        f'<span class="rank">#{rank} &middot; chars {r["start"]:,}–{r["end"]:,} &middot; score {r["score"]:.2f}</span></div>'
        f'<div class="passage">{body}</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Source details"):
        st.caption(f"File: `{r['file_path']}`  \nCharacter offsets: `[{r['start']}, {r['end']})` "
                   f"of {len(doc):,}")
        st.text(doc[max(0, r["start"] - 1200):r["end"] + 1200])


def render_abstention(reason: str) -> None:
    if reason.startswith("no indexed contract"):
        headline = "That contract isn't in the index"
        text = ("None of the indexed contracts matches the agreement named in your question. "
                "Choose a contract in the sidebar, or check the parties' names.")
    else:
        headline = "No clause answers this question"
        text = ("The closest passages don't read like an answer, so nothing is shown rather than a guess. "
                "The contract may not contain such a clause; try rephrasing to be sure.")
    st.markdown(f'<div class="empty"><h4>{headline}</h4><p>{text}</p>'
                f'<p class="detail">Reason: {html.escape(reason)}</p></div>', unsafe_allow_html=True)


def use_example() -> None:
    label = st.session_state.get("example")
    for name, contract_kw, question in EXAMPLES:
        if name == label:
            st.session_state.question = question
            st.session_state.contract = AUTO if contract_kw is None else next(
                t for t in st.session_state.contract_options if contract_kw.lower() in t.lower())
            st.session_state.run = True


def main() -> None:
    st.set_page_config(page_title="Clause Search", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    pipeline, titles, corpus = load()
    by_title = {t: fp for fp, t in titles.items()}
    st.session_state.contract_options = [AUTO] + sorted(by_title)

    with st.sidebar:
        st.selectbox("Contract", st.session_state.contract_options, key="contract",
                     help="Search one contract, or let the system find the contract named in your question.")
        k = st.slider("Passages to show", 1, 10, pipeline.config["top_k"])
        st.markdown(
            f'<p class="side-note">{len(corpus)} commercial contracts from CUAD '
            f'({sum(map(len, corpus.values())):,} characters).<br><br>'
            'Pipeline: match the contract by name, retrieve passages by meaning '
            '(bge-small embeddings), rerank with a cross-encoder, and return nothing '
            'when the best match is too weak.<br><br>'
            f'Settings: <code>{CONFIG_PATH}</code></p>', unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">Contract review</p><h1 class="title">Clause Search</h1>'
                '<p class="lede">Ask a question about a contract. You get the passages most likely to answer it, '
                'each with its exact location in the source document, or a clear "not found".</p>',
                unsafe_allow_html=True)

    with st.form("search", border=False):
        question = st.text_input("Question", key="question", label_visibility="collapsed",
                                 placeholder="e.g. Can this agreement be assigned without consent?")
        submitted = st.form_submit_button("Search", type="primary")
    st.pills("Examples", [e[0] for e in EXAMPLES], key="example", on_change=use_example,
             label_visibility="collapsed")

    if not (submitted or st.session_state.pop("run", False)) or not question.strip():
        return

    contract = None if st.session_state.contract == AUTO else by_title[st.session_state.contract]
    start = time.perf_counter()
    results, reason = pipeline.search(question, k, contract)
    elapsed = time.perf_counter() - start

    if reason:
        render_abstention(reason)
        return
    docs = {r["file_path"] for r in results}
    where = titles.get(next(iter(docs)), "") if len(docs) == 1 else f"{len(docs)} contracts"
    st.markdown(f'<p class="status">{len(results)} passages from {html.escape(where)} &middot; {elapsed:.2f}s</p>',
                unsafe_allow_html=True)
    for rank, r in enumerate(results, 1):
        render_result(rank, r, titles.get(r["file_path"], r["file_path"]), corpus)


main()
