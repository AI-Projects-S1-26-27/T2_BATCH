# Explainable Same-Event News Framing Analysis

CS F407 Artificial Intelligence, Team BATCH, Demo track
(Knowledge Representation & Reasoning + Applications of Deep Learning)

Given a headline, the system finds other outlets' headlines about the same event and explains
how their framing differs along four dimensions (information selection, actor/action framing,
evaluative wording, consequence emphasis), using logical facts, Horn rules and forward chaining.
Every difference is linked to the exact supporting words. No left/right labels are assigned.

## Setup

```bash
pip install -r requirements.txt
bash scripts/get_data.sh          # downloads BASIL into data/raw/BASIL (not committed)
python -m src.load                # S1: builds data/processed/stories.json + headlines.csv
python -m src.split               # fixed 6 dev / 14 test stories -> data/splits.json
python -m tests.test_load         # sanity checks
```

## Data (Stage S1)

BASIL (Fan et al., 2019): 100 events, each reported by Fox News, NYT and HuffPost
(2010–2019, 300 articles), with gold lexical/informational bias spans.

`data/processed/stories.json` holds a list of stories:

```
story_id, year, main_event, main_entities,
articles: { fox | nyt | hpo: {
    headline, headline_from_title, outlet, date, url, stance,
    sentences:  [{para, sent, text}],
    bias_spans: [{bias: lex|inf, polarity, target, text, para, sent, start, end}] } }
```

- Headlines come from the article title. Outlet suffixes such as "| Fox News" are removed.
  One NYT article has no title, so its first sentence (dateline removed) is used, as stated in the proposal.
- BASIL span offsets are relative to a single sentence, so the loader finds each span's sentence
  (1723 of the 1726 spans are located).
- `src.load.headline_pairs(story)` gives the 3 comparison pairs in a fixed order:
  (fox, nyt), (fox, hpo), (nyt, hpo). All later stages and baselines use this order.

## Structured extraction (Stage S2)

An LLM reads each headline and returns five lists, which later become logical facts:

| Field | Example (from "Police fire tear gas on peaceful protesters") | Becomes atom |
|---|---|---|
| actors | police (LawEnforcement), protesters (Activist) | used by Agent / Eval / IsA |
| items | "fire tear gas" | `Reports(h, i)` |
| agents | police → "fire tear gas" | `Agent(h, x, i)` |
| evals | "peaceful" about protesters, positive | `Eval(h, x, pol, w)` |
| outcomes | "curfew", type Safety | `Outcome(h, k, t)` |

Every extracted entry must quote words that literally occur in the headline. `validate()` drops
anything the model invents and stores character offsets, so the demo can highlight the evidence.
If nothing usable is extracted, or the API fails, the result has `ok: false` and a readable
`error` message instead of crashing (goal G3).

**Setup:** copy `.env.example` to `.env` and paste a Groq API key (free, from console.groq.com).

```bash
python -m src.extract --headline "Police fire tear gas on peaceful protesters"   # try one
python -m src.extract --split annotated    # the 20 dev+test stories (60 headlines)
python -m src.extract --split all          # all 300 headlines
python -m tests.test_extract               # offline tests, no key needed
```

- Results are cached in `data/extractions/cache.json`, keyed by prompt version, model and headline.
  Re-running only calls the API for new or failed headlines, so you can stop with Ctrl+C and resume.
- `data/extractions/extractions.json` maps `{story_id: {fox|nyt|hpo: result}}` for S3/S4.
  Both files are committed, so teammates and the demo don't need to call the API again.
- `src/llm.py` is a small client for any OpenAI-compatible API. Baseline B3 reuses it, so
  both use the same LLM, as the proposal requires.

## Cross-headline alignment (Stage S3)

Different headlines describe the same thing in different words ("fire tear gas" vs "use tear gas",
"Steele" vs "Michael Steele"). Alignment gives equivalent mentions one shared ID
(`i1`, `a1`, `o1`...), so the rules in S5 can tell what one headline has and another lacks.

For each story and each kind (items, actors, outcomes):
1. Embed every mention with Sentence-BERT (`all-MiniLM-L6-v2`).
2. For each headline pair, compute cosine similarity between their mentions.
3. Greedy one-to-one matching, most similar pair first, kept only if similarity ≥ threshold.
4. Merge matches into clusters (union-find). Two mentions from the same headline are never merged.

Actors also match when one name is contained in the other ("Steele" ~ "Michael Steele").
Each accepted match is stored with its similarity score, so the demo can show why two phrases were treated as the same.

**Threshold tuning on the 6 dev stories** (test stories are never used for tuning):
```bash
python -m src.align --dump-pairs   # data/alignment/dev_pairs.csv: fill "same" with y/n
python -m src.align --tune         # precision/recall/F1 per threshold -> data/alignment/params.json
python -m src.align                # align all extracted stories with the tuned thresholds
python -m src.align --show <story_id>   # inspect one story
python -m tests.test_align         # offline tests (stand-in embedder, no download)
```
Defaults before tuning: items 0.70, actors 0.75, outcomes 0.70.
Known limitation: embeddings ignore roles, so "police shoot man" and "man shoots police" can look similar.

## Logical facts and ontology (Stage S4)

`src/facts.py` turns one story's extractions and alignment into ground atoms, using the
shared IDs from S3:

```
Headline(h)  Pair(a,b)  Actor(h,x)  InstanceOf(x,category)  Reports(h,i)
Agent(h,x,i)  Eval(h,x,polarity,word)  Outcome(h,k,type)  IsA(c,d)  Specific(type)
```
Every atom keeps its provenance: the headline words and character offsets it came from.
The ontology is a small hand-written `IsA` hierarchy over actor categories
(e.g. LawEnforcement ⊑ Government ⊑ Institution ⊑ Agent) and outcome types (e.g. Health ⊑ Safety ⊑ Harm).

## Forward chaining and framing rules (Stage S5)

`src/engine.py` is a generic forward-chaining engine. Rules are Horn clauses written as data,
`NOT` is negation-as-failure under the closed-world assumption, and rules run in strata,
so a rule only negates predicates that are already complete (checked by `check_stratification`):

| Stratum | Rule | Clause |
|---|---|---|
| 0 | T | IsA(c,d) ∧ IsA(d,e) ⇒ IsA(c,e) |
| 0 | M | Actor(h,x) ⇒ Mentions(h,x);  Reports(h,i) ⇒ Mentions(h,i) |
| 0 | H | Eval(h,x,p,w) ⇒ HasEval(h,x,p) |
| 1 | R1 | Reports(A,i) ∧ ¬Reports(B,i) ⇒ InfoOnlyIn(A,B,i) |
| 1 | R2 | Agent(A,x,i) ∧ Reports(B,i) ∧ ¬Agent(B,x,i) ⇒ ActorShift(A,B,i) |
| 1 | R3 | Eval(A,x,p,w) ∧ Mentions(B,x) ∧ ¬HasEval(B,x,p) ⇒ EvalShift(A,B,x,w) |
| 1 | R4 | Outcome(A,k,t) ∧ Outcome(B,k′,t′) ∧ t≠t′ ∧ ¬IsA(t,t′) ∧ ¬IsA(t′,t) ⇒ OutcomeShift(A,B,t,t′) |

R3 and R4 depend on facts derived in stratum 0 (Mentions, HasEval, and the IsA closure), so the chaining is multi-step.
Every derived difference keeps a trace: the rule, the atoms it matched, and the atoms checked to be absent.
That trace, together with the headline evidence, is the explanation.
For R1, the provenance step checks whether B's *article* expresses the item (best Sentence-BERT
match among the article's sentences, threshold 0.55). If it does, the result is marked as a headline-level omission and that sentence is cited.
Symmetric R4 results are reported once.

```bash
python -m src.facts --story-id <id>     # print a story's atoms
python -m src.engine --story-id <id>    # explanations + rule traces for one story
python -m src.engine                    # all extracted stories -> data/differences/differences.json
python scripts/export_review.py         # CSV of candidate differences for evidence-support rating
python -m tests.test_engine             # engine + rules tests
```

## Baselines (M7)

All three baselines return difference records in the same format as the main system
(`dimension`, `from`, `to`, cited `evidence` with offsets, `explanation`), so `eval/` scores every system the same way.

| ID | Method | Question (proposal 5.4) |
|---|---|---|
| B1 | TF-IDF word overlap: content words of A whose stem never occurs in B are flagged (IDF from all 300 BASIL headlines); adjacent flagged words merge into phrases | Can word overlap alone find framing differences? |
| B2 | Sentence-BERT: each headline is split into short phrases (runs of content words, ≤3 words); a phrase of A whose best cosine similarity to B's phrases is below the S3 item threshold is flagged | Is semantic matching enough without structured fields and rules? |
| B3 | Direct LLM (same model as S2): both headlines + the first 4 sentences of each article; the model lists differences in the 4 dimensions and cites headline words | Do structure and explicit rules add value over direct prompting? |

B1 and B2 cannot tell actors, tone or consequences apart. For scoring, a flagged phrase is labelled
D3 if it contains a word from a small fixed lexicon of loaded words (`baselines/common.py`), and D1 otherwise.
They never predict D2 or D4. B3's cited words are checked against the headlines, and a citation that
isn't found is kept but marked `evidence_found: false`. B3 answers are cached in `data/baselines/b3_cache.json`.

```bash
python -m baselines.run                    # B1+B2+B3 on the 20 annotated stories -> data/baselines/B*.json
python -m baselines.run --systems B1 B2    # no API calls
python -m baselines.run --show <story_id>  # print one story's outputs from all three
python -m tests.test_baselines             # offline tests
```

## Evaluation

`eval/evaluate.py` scores every system against the gold labels. The unit is one headline pair ×
one dimension: a system detects dimension *d* for pair {A, B} if it outputs at least one record of
that dimension between them, in either direction. Output is precision / recall / F1 per dimension, plus micro and macro F1.

- **Systems:** OURS (`data/differences/differences.json`), B1–B3 (`data/baselines/`), and
  **ABLATION**: S2 + S3 with no rules, where a dimension is flagged whenever the two headlines' extracted fields differ (proposal claim C3).
- **Evidence overlap:** for true positives, whether the cited words share a content word with the annotator's words.
- **Evidence-support rate:** a blind, shuffled rating sheet (up to 40 outputs per system) rated yes/partial/no.

```bash
python -m src.engine                       # our system's outputs
python -m baselines.run                    # B1-B3 outputs
python -m eval.evaluate                    # test split -> eval/results/results_test.md/.json, pairs_test.csv
python -m eval.evaluate --split dev
python -m eval.evaluate --make-rating-sheet    # eval/results/evidence_rating_test.csv (+ _KEY.csv, do not open)
python -m eval.evaluate --score-ratings        # after the sheet is filled
python -m tests.test_evaluate
```

## Web demo (Stage S6)

```bash
streamlit run app.py          # opens http://localhost:8501
```

Three tabs:
1. **Prepared BASIL stories.** Results for the 20 annotated stories come from the cache, so they are instant and need no API key.
   ★ marks the five demo stories (one per framing dimension, plus mixed cases). Other BASIL stories are extracted live.
2. **Analyse a new headline.** Same-event retrieval (`src/retrieve.py`) finds the BASIL event, and your headline is
   compared with each outlet's. If there is no same-event match, the demo says so and does not compare unrelated headlines.
3. **Compare your own headlines.** Paste 2–3 headlines about one event (e.g. today's news), which are analysed live.

Each result starts with an **Overview**: key takeaways, a framing profile per outlet (what only it reports,
what it leaves out, who it shows acting, loaded words, consequences stressed), and an outlet × dimension table.
It is derived by extra stratified rules in `src/summary.py`, e.g. `Exclusive(h,i) <= Reports(h,i) AND NOT ReportedByOther(h,i)`
and `Common(i) <= Reports(h,i) AND NOT Missing(i)`, so every statement traces back to facts.
Below it, for every pair, the evidence words are highlighted in their dimension's colour. Each difference shows its explanation,
the rule that fired, the matched facts, the facts checked absent, and any article sentence. A second tab shows B1–B3 side by side,
and an expander lists the extracted logical facts. Every failure (no key, API error, unusable extraction, no match)
produces a readable message instead of a crash (`src/pipeline.py`).

**Retrieval:** TF-IDF over each BASIL article's headline plus its first 4 sentences. A match needs score ≥ cutoff and
at least 2 shared content words. The cutoff (0.11) was chosen on the dev + unused stories (`python -m src.retrieve --tune`):
the correct story ranks first for 90.7% of headlines, and the cutoff accepts 74.8% of true same-event queries while rejecting 88.0% of no-match queries.

```bash
python -m src.retrieve --query "Senate overrides Obama veto of 9/11 bill"
python -m tests.test_pipeline
```

### Deploying on Streamlit Community Cloud (free)
1. Push the repo to GitHub (public), then sign in at share.streamlit.io with GitHub.
2. Choose **Create app → Deploy a public app from GitHub**: repository `spiritual-drink27/AI_PROJECT`, branch `main`,
   main file `app.py`. Under **Advanced settings**, pick Python 3.11 and paste into **Secrets**:
   ```toml
   LLM_API_KEY = "your-groq-key"
   ```
3. Click Deploy. The first build installs PyTorch (CPU) and takes several minutes. Without the secret, the prepared stories still work.

## Gold annotation (evaluation data)

20 stories (60 headline pairs) are annotated by hand, as set out in proposal Section 5.1.
For each pair, the annotator marks which of the four framing differences are present
(D1 information, D2 actor/action, D3 evaluative wording, D4 consequence emphasis)
and copies the words that show them. The guidelines are in `eval/annotation/GUIDELINES.txt`
and on the second sheet of every workbook.

| Split | Annotators | Files |
|---|---|---|
| dev (6 stories) | M3 Sameeksha 3, M4 Zainab 3 | `eval/annotation/dev/` |
| test (14 stories) | M1 Siddharth 3, M6 Sachin 3, M7 Nitya 4, M8 Megha 4 | `eval/annotation/test/` |

Five test stories get a second, independent annotator for Cohen's κ (`eval/annotation/assignment.json`).
**M2–M5 must not open `eval/annotation/test/` or `eval/gold/gold_test.csv` before the single test run.**

```bash
python scripts/make_annotation_sheets.py    # (already run) creates the workbooks
python -m eval.merge_annotations             # progress + gold_dev/gold_test.csv + kappa.json + disagreements.csv
python -m eval.merge_annotations --split dev # dev only (safe for M2-M5)
python -m tests.test_annotation              # simulated-annotator test
```
M8 adjudicates by filling the `final` column of `eval/gold/disagreements.csv`, then re-runs the merge.
Gold labels come from the first annotator, with M8's adjudicated decisions replacing them where filled in.

## Repository layout

| Path | Stage | Owner |
|---|---|---|
| `src/load.py`, `src/split.py` | S1 data loading, grouping, dev/test split | M1, M2 |
| `src/extract.py` | S2 LLM structured extraction | M2 |
| `src/align.py` | S3 Sentence-BERT alignment | M3 |
| `src/facts.py` | S4 ontology and logical facts | M4 |
| `src/engine.py` | S5 forward chaining and rules R1–R4 | M5 |
| `src/engine.py` (`explain()`), `src/summary.py`, `src/pipeline.py`, `app.py` | S6 explanations and web demo | M6 |
| `baselines/` | B1 TF-IDF, B2 Sentence-BERT, B3 direct LLM | M7 |
| `eval/` | annotation, metrics, ablation | M8 |
