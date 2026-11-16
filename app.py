"""Web demo (S6): explainable same-event news framing analysis.

Run locally:   streamlit run app.py
Three ways in:
  1. Prepared BASIL stories: cached results, instant (demo backbone)
  2. A new headline: retrieved against BASIL; compared with that event's coverage
  3. Your own headlines: 2-3 headlines about one event, analysed live
"""

import html
import json
import os

import streamlit as st

st.set_page_config(page_title="Same-Event Framing Analysis", page_icon="📰", layout="wide")

# Streamlit Cloud: the API key comes from the app's Secrets instead of a .env file
try:
    for _k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

from src.engine import explain                                    # noqa: E402
from src.load import SOURCES, SOURCE_NAMES, load_stories          # noqa: E402
from src.pipeline import Tools, analyze, by_pair, clean_name      # noqa: E402

DIM_LABEL = {"D1_info": "Information selection", "D2_actor": "Actor / action framing",
             "D3_eval": "Evaluative wording", "D4_outcome": "Consequence emphasis"}
DIM_COLOR = {"D1_info": "#cfe2ff", "D2_actor": "#ffe0b3", "D3_eval": "#f8c8d4", "D4_outcome": "#c9ecd4"}
DIM_RULE = {"D1_info": "R1", "D2_actor": "R2", "D3_eval": "R3", "D4_outcome": "R4"}
# Prepared demo stories: one per dimension plus mixed cases (all cached)
FEATURED = ["38f7cbb7-5d6a-4c89-bcbd-8e164144172a", "c6a0e65c-4f8b-41de-8200-cd700f683c64",
            "576116e5-2b18-4d35-be73-36f5863c9468", "a28a0329-0887-42dc-bd3d-3bb4f8926757",
            "1b96a5db-8974-46d3-82d7-5dfef02f650e"]


# ------------------------------------------------------------------ cached resources

@st.cache_data
def stories():
    return {s["story_id"]: s for s in load_stories()}


@st.cache_data
def cached_extractions():
    try:
        with open(os.path.join("data", "extractions", "extractions.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_resource(show_spinner="Loading Sentence-BERT (first time only)...")
def tools():
    from baselines.b3_llm import B3
    from baselines.run import load_b1_model
    from src.align import SBERTEmbedder, load_thresholds
    from src.extract import Extractor
    return Tools(extractor=Extractor(), embedder=SBERTEmbedder(), b1_model=load_b1_model(),
                 b3=B3(), thresholds=load_thresholds())


@st.cache_resource
def retriever():
    from src.retrieve import Retriever
    return Retriever(list(stories().values()))


def has_key():
    from src.llm import load_dotenv
    load_dotenv()
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"))


def outlet(src):
    return SOURCE_NAMES.get(src, "Your headline" if src == "you" else src.title())


# ------------------------------------------------------------------ rendering helpers

def highlight(text, spans):
    """HTML of a headline with evidence spans marked in their dimension's colour."""
    spans = sorted((s for s in spans if s[0] is not None and s[1] is not None and s[1] > s[0]),
                   key=lambda s: (s[0], -s[1]))
    out, pos = [], 0
    for start, end, dim in spans:
        if start < pos:
            continue                                   # skip overlaps
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark style="background:{DIM_COLOR.get(dim, "#eee")};color:#111;'
                   f'padding:0 2px;border-radius:3px">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def spans_for(records, src):
    return [(e.get("start"), e.get("end"), r["dimension"])
            for r in records for e in r.get("evidence", []) if e.get("source") == src]


def legend():
    st.markdown(" ".join(f'<span style="background:{DIM_COLOR[d]};color:#111;padding:2px 6px;'
                         f'border-radius:4px;margin-right:6px">{DIM_RULE[d]} · {DIM_LABEL[d]}</span>'
                         for d in DIM_LABEL), unsafe_allow_html=True)


def render_difference(r, n):
    color = DIM_COLOR[r["dimension"]]
    st.markdown(f'<div style="border-left:6px solid {color};padding:4px 10px;margin:6px 0">'
                f'<b>{DIM_RULE[r["dimension"]]} · {DIM_LABEL[r["dimension"]]}</b><br>'
                f'{html.escape(explain(r))}</div>', unsafe_allow_html=True)
    with st.expander("Why? Rule trace and evidence", expanded=False):
        for t in r.get("trace", [])[:2]:
            st.markdown(f"**Rule {t['rule']}:** `{t['rule_text']}`")
            st.markdown("**Matched facts:** " + ", ".join(f"`{m}`" for m in t["matched"]
                                                          if not m.startswith("Pair")))
            if t["absent"]:
                st.markdown("**Checked absent (closed-world):** " + ", ".join(f"`{a}`" for a in t["absent"]))
        cited = [f"{outlet(e['source'])}: “{e['text']}”" for e in r.get("evidence", []) if e.get("text")]
        if cited:
            st.markdown("**Headline evidence:** " + "; ".join(html.escape(c) for c in cited))
        if r.get("article_evidence"):
            ae = r["article_evidence"]
            st.markdown(f"**{outlet(ae['source'])} article also says it** "
                        f"(similarity {ae['score']}): _{html.escape(ae['sentence'][:300])}_")
        elif r.get("status") == "not_found_in_article":
            st.caption("Not found in the other outlet's article either.")


def render_baselines(result, pair):
    cols = st.columns(3)
    notes = {"B1": "TF-IDF: words with no match in the other headline",
             "B2": "Sentence-BERT: phrases with no similar phrase",
             "B3": "Direct LLM comparison"}
    for col, name in zip(cols, ("B1", "B2", "B3")):
        with col:
            st.markdown(f"**{name}**  \n<small>{notes[name]}</small>", unsafe_allow_html=True)
            recs = by_pair(result["baselines"].get(name), [pair]).get(frozenset(pair), [])
            if name not in result["baselines"]:
                st.caption("not run (needs an API key)" if name == "B3" else "not available")
            elif not recs:
                st.caption("no differences reported")
            for r in recs[:12]:
                ev = r["evidence"][0]
                flag = "" if r.get("evidence_found", True) else " ⚠️ cited words not in headline"
                st.markdown(f"- `{DIM_RULE.get(r['dimension'], r['dimension'])}` "
                            f"{outlet(r['from'])}: “{html.escape(ev.get('text') or '')}”{flag}")
            if len(recs) > 12:
                st.caption(f"... and {len(recs) - 12} more")


def render_facts(result):
    kb = result.get("kb")
    if not kb:
        st.caption("No facts (extraction failed).")
        return
    from src.facts import atom_key
    labels = kb["labels"]
    for src in result["headlines"]:
        rows = [a for a in kb["atoms"] if len(a) > 1 and a[1] == src and a[0] not in ("Pair", "Headline")]
        st.markdown(f"**{outlet(src)}**")
        st.code("\n".join(f"{atom_key(a):45s} " + " ".join(f"[{labels[x]}]" for x in a[2:] if x in labels)
                          for a in rows) or "(none)", language=None)


def render_result(result, focus=None):
    """focus: only show pairs that include this source (e.g. 'you')."""
    for e in result["errors"]:
        st.warning(e)
    if result["message"]:
        (st.info if result["ok"] else st.error)(result["message"])
    if not result["headlines"]:
        return
    legend()
    groups = by_pair(result["differences"], result["pairs"])
    for pair in result["pairs"]:
        a, b = pair
        if focus and focus not in pair:
            continue
        recs = groups[frozenset(pair)]
        st.markdown("---")
        st.subheader(f"{outlet(a)} vs {outlet(b)}")
        c1, c2 = st.columns(2)
        for col, src in ((c1, a), (c2, b)):
            with col:
                st.caption(outlet(src))
                st.markdown(f"<div style='font-size:1.15rem'>"
                            f"{highlight(result['headlines'][src], spans_for(recs, src))}</div>",
                            unsafe_allow_html=True)
        t1, t2 = st.tabs([f"Our system ({len(recs)} differences)", "Baselines side by side"])
        failed = [x for x in pair if not result["extractions"].get(x, {}).get("ok")]
        with t1:
            if failed:
                st.caption(f"{' and '.join(outlet(x) for x in failed)} could not be analysed "
                           "(see the warning above), so the rules cannot compare this pair.")
            elif not result["ok"]:
                st.caption("Rule-based comparison unavailable for this input.")
            elif not recs:
                st.caption("No differences derived for this pair under rules R1-R4.")
            for n, r in enumerate(sorted(recs, key=lambda r: (r["dimension"], r["from"]))):
                render_difference(r, n)
        with t2:
            render_baselines(result, pair)
    with st.expander("Extracted logical facts (S4)"):
        render_facts(result)


# ------------------------------------------------------------------ page

st.title("📰 Same-event headline framing")
st.markdown("Compares how different outlets **frame the same event**: what information they include, "
            "who they show acting, which loaded words they use, and which consequences they stress. "
            "Every difference comes from an explicit logical rule and points to the exact words. "
            "*This is framing analysis, not a political judgment: no outlet is labelled left or right.*")

with st.sidebar:
    st.header("About")
    st.markdown("**Pipeline:** LLM extraction → Sentence-BERT alignment → logical facts + ontology → "
                "forward chaining with rules R1–R4 → evidence-linked explanations.")
    st.markdown("**Data:** BASIL (Fan et al., 2019): 100 events covered by Fox News, NYT and HuffPost.")
    st.markdown("LLM key: " + ("✅ configured (live analysis on)" if has_key() else
                               "❌ not configured: prepared stories still work; live analysis is off"))
    st.caption("CS F407 Artificial Intelligence · Team BATCH")

tab1, tab2, tab3 = st.tabs(["Prepared BASIL stories", "Analyse a new headline", "Compare your own headlines"])

with tab1:
    all_st = stories()
    ext = cached_extractions()
    ids = FEATURED + sorted(s for s in ext if s not in FEATURED) + \
        sorted(s for s in all_st if s not in ext)

    def label(sid):
        s = all_st[sid]
        tag = "★ " if sid in FEATURED else ("" if sid in ext else "(live) ")
        return f"{tag}{s['year']} · {s.get('main_event') or s['articles']['nyt']['headline']}"[:120]

    sid = st.selectbox("Choose a story", ids, format_func=label)
    arts = all_st[sid]["articles"]
    heads = {s: arts[s]["headline"] for s in SOURCES}
    sents = {s: [x["text"] for x in arts[s]["sentences"]] for s in SOURCES}
    if sid not in ext and not has_key():
        st.info("This story has no cached results and no LLM key is configured. Pick a ★ story.")
    elif sid in ext or st.button("Analyse story (live extraction, about 10 s)", type="primary"):
        with st.spinner("Analysing..."):
            res = analyze(heads, tools(), articles=sents, extractions=ext.get(sid))
        render_result(res)

with tab2:
    q = st.text_input("Type a news headline", placeholder="e.g. Senate overrides Obama's veto of 9/11 bill")
    if q:
        hit = retriever().search(q)
        if not hit["results"] or not hit["match"]:
            st.info("**No same-event match found in the BASIL collection.** The system only compares "
                    "headlines about the same event, so it will not compare this headline with unrelated "
                    "coverage. Try the *Compare your own headlines* tab to paste other outlets' "
                    "headlines about this event.")
            if hit["results"]:
                best = hit["results"][0]
                st.caption(f"Closest candidate (score {best['score']}, cutoff {hit['cutoff']}; shared words: "
                           f"{', '.join(best['shared_words']) or 'none'}): "
                           f"“{best['headlines'][best['best_source']]}”")
        else:
            best = hit["results"][0]
            st.success(f"Same event found (TF-IDF score {best['score']} ≥ {hit['cutoff']}; shared words: "
                       f"{', '.join(best['shared_words'])}): **{best['main_event'] or best['story_id']}**")
            story = stories()[best["story_id"]]
            heads = {"you": q, **{s: story["articles"][s]["headline"] for s in SOURCES}}
            sents = {s: [x["text"] for x in story["articles"][s]["sentences"]] for s in SOURCES}
            if not has_key():
                st.error("Live analysis needs an LLM key (your headline must be extracted).")
            else:
                show_all = st.checkbox("Also show comparisons between the stored outlets")
                with st.spinner("Extracting, aligning and reasoning..."):
                    res = analyze(heads, tools(), articles=sents,
                                  extractions=cached_extractions().get(best["story_id"]))
                render_result(res, focus=None if show_all else "you")

with tab3:
    st.markdown("Paste two or three headlines about **the same event** from different outlets "
                "(e.g. today's news). Outlet names are optional.")
    rows = []
    for n in range(3):
        c1, c2 = st.columns([1, 4])
        name = c1.text_input("Outlet", key=f"o{n}", placeholder=f"Outlet {n + 1}")
        text = c2.text_input("Headline", key=f"h{n}", placeholder="Headline" + (" (optional)" if n == 2 else ""))
        rows.append((name, text))
    if st.button("Compare", type="primary"):
        used, heads = set(), {}
        for n, (name, text) in enumerate(rows):
            if text.strip():
                key = clean_name(name or f"outlet{n + 1}", used)
                used.add(key)
                heads[key] = text
        if len(heads) < 2:
            st.error("Please enter at least two headlines.")
        elif not has_key():
            st.error("Live analysis needs an LLM key. Prepared stories still work in the first tab.")
        else:
            with st.spinner("Extracting, aligning and reasoning..."):
                res = analyze(heads, tools())
            render_result(res)
