# TeleKG-Agent

**A Knowledge Graph-Driven Agentic Knowledge Plane for Autonomous Telecom Test Generation, KPI Translation, and Signal-Native Foundation Models.**

A prototype exploring three connected ideas in telecom AI:

1. **Dynamic, change-driven test generation** — instead of running a static test suite after every software release or config change, traverse a knowledge graph to figure out exactly what's at risk and generate only the tests that matter.
2. **Natural-language KPI translation** — let an engineer type "percentage of dropped calls" and have the system map that to a real, auditable formula over real PM counters, instead of hardcoding KPI definitions.
3. **Signal-native pretraining** — a small proof-of-concept exploring whether a model pretrained directly on telecom time-series (not text) learns better representations than one trained from scratch on a downstream task.

This is a personal research prototype, built solo, with no GPU access and no access to real operator data. Everything here runs on synthetic data, on a single CPU. Where that matters for interpreting results, it's called out explicitly rather than glossed over.

---

## Why this exists

A 5G network changes constantly: software releases, configuration tweaks, new slices, cell outages. Each change can degrade KPIs (Accessibility, Throughput, Latency...) through causal chains that aren't written down anywhere machine-readable. Current practice is to re-run a full static test suite after every change, which is slow and doesn't actually target what changed.

The idea here: model the network — its topology, its KPIs, the causal relationships between KPIs, which features affect which KPIs, and which tests validate which KPIs — as a single knowledge graph. Then a small pipeline of agents can traverse that graph to answer questions like "this software release touches the Energy Saving feature — which KPIs are at risk, which cells, and which tests do we actually need to run?" without re-running everything.

That graph also turns out to be useful for a second problem: KPIs are not primitive measurements, they're formulas over PM counters (e.g. `Accessibility = RRC_Setup_Success / RRC_Setup_Attempt * 100`). If you expose that counter substrate, you can let someone type a KPI definition in plain English and have an LLM — grounded in retrieval over the real counter dictionary, so it can't invent counter names — translate it into a real, computable formula.

The third piece is more speculative: every "telecom LLM" that exists today (TelecomGPT, TSLAM, SoftBank's LTM) is trained on telecom *text* — specs, RFCs, tickets. None are pretrained directly on telecom *signal* data — KPI/counter time-series — the way a vision model is pretrained on pixels. This repo includes a small (110K-parameter) proof of concept testing whether that's worth doing, at the only scale available without a GPU.

---

## Architecture

```
Change Event (SW_UPGRADE / CONFIG_CHANGE / CELL_OUTAGE / SLICE_CREATE)
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                      TeleKG (FalkorDB graph)                   │
│                                                                  │
│  Layer 1 — Topology       Cell ⇄ DU ⇄ K8sCluster ⇄ Region       │
│  Layer 2 — KPI causality  KPI --[IMPACTS, weighted]--> KPI       │
│  Layer 3 — Feature/test   SoftwareRelease -> Feature -> KPI     │
│                            Feature -> TestCase -> KPI            │
│                                                                  │
│  PM Counter substrate (telekg/pm_registry.py)                   │
│  KPI = formula(counters), e.g.                                  │
│    Accessibility = RRC_Setup_Success / RRC_Setup_Attempt * 100  │
└────────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
┌──────────────────────┐      ┌─────────────────────────────────┐
│  5-Agent Pipeline      │      │  KPI Translator (RAG)            │
│  (agents/pipeline.py)  │      │  (telekg/kpi_translator.py)      │
│                        │      │                                   │
│  NetworkUnderstanding  │      │  free text -> retrieve candidate │
│  → ImpactAnalysis      │      │  counters -> LLM composes/matches│
│  → RootCause           │      │  formula -> validate (no         │
│  → TestGeneration      │      │  hallucinated counters allowed)  │
│  → Verification        │      │  -> compute                       │
└──────────────────────┘      └─────────────────────────────────┘
        │
        ▼
  Generated, prioritised pytest test cases
```

A separate, standalone piece (not wired into the graph) is **TeleSignal-Tiny**
(`telekg/telesignal_model.py`), a small transformer pretrained directly on
multivariate KPI time-series via masked-patch reconstruction, evaluated on
downstream anomaly detection. See [Signal-Native Foundation Models](#part-3-signal-native-foundation-models-telesignal-tiny) below.

**Design principle throughout:** the graph holds all structural/causal
knowledge; the LLM is only ever asked to do two things — turn a structured
graph result into a sentence, or turn a structured spec into code. It is
never asked to recall a telecom fact from training data. This is what makes
the system's output traceable: every generated test or translated KPI can be
walked back through the exact graph path that produced it.

---

## 30-Second Demo (start here)

Before any setup, there's a single zero-dependency file you can run right
now to see the core causal reasoning in action:

```bash
python demo_telecom_graph.py --list
python demo_telecom_graph.py --kpi throughput
```

This is a small, hand-written, in-memory version of the same KPI causality
graph used by `telekg/reasoner.py`'s root-cause traversal, stripped of
FalkorDB, LangGraph, and Groq so it runs anywhere with nothing but the
Python standard library. It exists for live demos, interviews, and coding
rounds where spinning up infrastructure first isn't an option. Type a KPI
name and it walks the full causal chain backward to root causes, complete
with weighted evidence and a recommended remedy, the same reasoning
`telekg/reasoner.py`'s `root_cause()` does against the real FalkorDB graph,
just without needing the real graph to see it work.

For the real system (full FalkorDB graph, 5-agent LangGraph pipeline, Groq
narration, PM counter substrate), continue to the Quick Start below.

---

## Quick Start

```bash
git clone <this repo>
cd telekg-agent
pip install -r requirements.txt   # see below — most of this is optional
```

### Zero-dependency demo (no FalkorDB, no LLM, no API key)

```bash
python main.py --dry-run --eval
```

This runs the full 5-agent pipeline against an in-memory mock graph and a
rule-based stub instead of an LLM. It's enough to see the architecture work
end to end — impact analysis, root cause narrative (templated), test
generation (templated), and the baseline-vs-dynamic test selection
evaluation — in under a second, no setup required.

### With a real LLM (Groq)

[Groq](https://console.groq.com) gives you a free API key and very fast
inference — no local GPU, no Ollama, no model download.

```bash
export GROQ_API_KEY=gsk_...
pip install groq
python main.py --eval
```

Without `--dry-run`, the pipeline will use Groq if `GROQ_API_KEY` is set,
and falls back to the offline stub (with a printed warning) if it's missing
or a call fails — it never crashes the pipeline.

### With a real graph (FalkorDB)

```bash
docker run -p 6379:6379 falkordb/falkordb:latest
pip install falkordb
python main.py --eval   # builds the graph in FalkorDB instead of dry-run mode
```

---

## Repository Structure

```
telekg/
  schema.py            — Node/edge types, KPI ontology, feature→KPI map
  pm_registry.py        — PM counter dictionary + KPI formula registry
                           (the ground truth the KPI translator is checked against)
  pm_simulator.py        — Generates realistic PM counter values and
                           multivariate time-series from those formulas
  simulator.py           — Synthetic network generator (100 cells, change events)
  graph_builder.py       — Builds the 3-layer graph in FalkorDB
  reasoner.py            — Graph traversal operations the agents call
                           (impact analysis, root cause chains, coverage gaps)
  kpi_translator.py      — RAG-based NL → formula → value KPI translator
  kpi_eval.py             — 17-query benchmark with ground truth for the translator
  telesignal_model.py    — Small signal-native transformer (pretraining + probing)
agents/
  pipeline.py             — 5-agent LangGraph pipeline + Groq LLM wrapper
main.py                   — Demo runner: builds graph, runs pipeline, prints report
demo_telecom_graph.py      — Zero-dependency standalone causal-graph demo (see above)
run_kpi_eval.py            — Run the KPI translation benchmark against real Groq
run_telesignal_poc.py      — Run the signal-native pretraining proof of concept
data/                      — Generated synthetic data (gitignored by default)
```

---

## Part 1: Dynamic Test Generation

### The graph

`telekg/schema.py` defines the ontology. Three layers, one graph:

- **L1 Topology** — `Cell`, `DistributedUnit`, `K8sCluster`, `Region`,
  `NetworkSlice`, connected by `SERVED_BY`, `HOSTED_ON`, `PART_OF_REGION`.
- **L2 KPI causality** — `KPI` nodes connected by weighted `IMPACTS` edges,
  e.g. `PRB_Utilization -[IMPACTS, weight=0.85]-> Throughput`. These
  weights are derived from the PM counter formulas and 3GPP references in
  `pm_registry.py`, not invented.
- **L3 Feature/software/test** — `SoftwareRelease -[AFFECTS_FEATURE]->
  Feature -[AFFECTS_KPI]-> KPI`, and `Feature -[COVERED_BY]-> TestCase
  -[VALIDATES]-> KPI`.

A single Cypher traversal from a `SoftwareRelease` node — implemented in
`telekg/reasoner.py: release_impact()` — returns the complete risk
picture: which features are touched, which KPIs are directly and
indirectly (via the causal graph) at risk, which cells are affected, and
which tests are required.

### The agents

`agents/pipeline.py` implements five agents as a LangGraph state machine
(or a plain sequential loop if LangGraph isn't installed — both code paths
exist and are tested):

| Agent | What it does | Uses LLM? |
|---|---|---|
| `NetworkUnderstandingAgent` | Ingests a change event, writes it to the graph | No |
| `ImpactAnalysisAgent` | Traverses the graph for affected features/KPIs/cells/tests | Yes — narrates the result |
| `RootCauseAgent` | Traces the causal KPI chain backward | Yes — narrates the result |
| `TestGenerationAgent` | Generates pytest code for required tests | Yes — generates code |
| `VerificationAgent` | Queues tests for execution (stub) | No |

Only `TestGenerationAgent`'s use of the LLM is load-bearing — turning a
structured test spec into actual code is genuine language-to-code
generation. The narration calls in the other two agents are nice-to-have
(they turn graph output into readable sentences) and degrade cleanly to a
shorter, less fluent stub if no LLM is configured.

### Running it

```bash
python main.py --dry-run --eval
```

prints, in order: the synthetic dataset stats, the agent execution log,
the impact analysis result, the root cause narrative, the generated test
cases, and an evaluation comparing a static "run everything" baseline
against the graph-guided dynamic selection. On the bundled synthetic
dataset this reduces the active test suite by ~72% for a representative
software release while keeping all critical-priority tests.

---

## Part 2: KPI Translation (RAG)

### The problem

KPIs aren't raw measurements — they're formulas over PM counters. The
`telekg/pm_registry.py` module is the ground truth: 28 PM counters (with
3GPP/ETSI references) and 10 KPI formulas expressed as real ratios over
them, e.g.:

```python
"Accessibility": formula = RRC_SETUP_SUCCESS / RRC_SETUP_ATTEMPT * 100
"Latency":       formula = RTT_SUM_MS / RTT_SAMPLE_COUNT
```

`telekg/pm_simulator.py` generates raw counter values that round-trip
through these formulas to realistic KPI values (verified to ~0% error in
the consistency check at the bottom of that file).

### The translator

`telekg/kpi_translator.py` implements the actual pipeline:

1. **Retrieve** — embed the user's free-text KPI definition (e.g. *"what
   fraction of connection attempts actually succeed?"*) and retrieve the
   top-k most relevant PM counters by semantic similarity
   (`sentence-transformers`, with a keyword-overlap fallback if that's
   not installed).
2. **Generate** — give the LLM the retrieved candidates and ask it to
   either match an existing registered KPI or compose a new formula
   *using only the retrieved counter names*.
3. **Validate** — reject the response if it references any counter not in
   the registry (no hallucinated counters allowed) or if the formula
   isn't evaluable.
4. **Compute** — evaluate the validated formula against real counter
   values.

This is genuinely retrieval-*augmented*: the LLM can't free-associate a
plausible-sounding counter name, it can only select from what retrieval
actually surfaced as a real registry entry.

### Evaluating it

`telekg/kpi_eval.py` is a 17-query benchmark with ground truth: 12
paraphrases that should match an existing registered KPI, 4 that require
composing a genuinely new formula, and 1 deliberately vague query to check
the system doesn't confidently hallucinate when it shouldn't.

```bash
export GROQ_API_KEY=gsk_...
pip install groq sentence-transformers
python run_kpi_eval.py --json results.json
```

This reports retrieval accuracy, KPI-match accuracy, composed-formula
validity rate, and numeric error against ground truth. Running
`python -m telekg.kpi_eval` directly (no Groq key) runs the same benchmark
against the offline rule-based fallback, which is intentionally weak — it
exists to prove the harness itself discriminates between a good and bad
translator, not as a real baseline to report.

---

## Part 3: Signal-Native Foundation Models (TeleSignal-Tiny)

### The question

Every published telecom-specific LLM (TelecomGPT, TSLAM, SoftBank's LTM,
and others) adapts a general-purpose *text* LLM via continual pretraining
or fine-tuning on telecom documents — specs, RFCs, tickets. None pretrain
directly on telecom *signal* data: KPI and PM-counter time-series, the way
the network itself actually generates information. That's a real,
citable gap in the literature, not an assumption — see the comments at
the top of `telekg/telesignal_model.py` for the specific papers checked.

### The proof of concept

`telekg/telesignal_model.py` implements a small (110,864-parameter)
transformer that treats KPI time-series as its native modality:

- Multivariate KPI series (10 channels) are split into non-overlapping
  8-timestep patches.
- Each patch is linearly projected into a 64-dim embedding — the signal
  equivalent of a word embedding.
- Pretraining objective: mask ~40% of patches, train the model to
  reconstruct the masked values from context. This is BERT's masked
  language modelling objective, applied to numbers instead of words
  (same family as PatchTST-style time-series foundation models).

`run_telesignal_poc.py` compares three conditions, all using the
*identical* architecture and downstream task (binary anomaly detection on
a held-out window):

| Condition | What it tests |
|---|---|
| A — Random init, frozen probe | The floor: no domain pretraining at all |
| B — Pretrained, frozen probe | Do the learned representations alone transfer? |
| C — Pretrained, fine-tuned | The realistic deployment setting |

```bash
pip install torch
python run_telesignal_poc.py --n-cells 60 --n-timesteps 1500
```

Runs in ~2 minutes on a single CPU core.

### Honest results (3 seeds, 90,000 observations, CPU)

| Condition | F1 (mean ± std) |
|---|---|
| A: Random init, frozen | 0.663 ± 0.104 |
| B: Pretrained, frozen | 0.777 ± 0.081 |
| C: Pretrained, fine-tuned | **0.981 ± 0.010** |

The fine-tuned result (C) is robust and consistent across seeds — clear
evidence that signal-native pretraining produces a useful initialisation.
The frozen-probe comparison (B vs A) is directionally positive but noisier
at this scale (one of three seeds showed no improvement at all) — reported
as suggestive, not conclusive, evidence.

**This is a proof of concept, not a competitor to any industrial telecom
foundation model effort.** The model is 2-3 orders of magnitude smaller
than even a modest production model, trained on synthetic data, on one
CPU. What it does establish: the architectural question ("is signal-native
pretraining worth doing at all?") is answerable, even without GPU access,
and the answer at this scale leans yes for the realistic (fine-tuned)
deployment setting.

---

## Dependencies and What's Actually Optional

```
numpy                   — required everywhere
falkordb                — optional; dry-run mode uses an in-memory mock instead
groq                     — optional; offline rule-based stub used without it
langgraph                — optional; sequential fallback runner used without it
sentence-transformers    — optional; keyword-overlap retrieval used without it
torch                    — only needed for telesignal_model.py / run_telesignal_poc.py
```

The whole repository is designed so that `python main.py --dry-run --eval`
and `python -m telekg.kpi_eval` work with just `numpy` installed, and
`demo_telecom_graph.py` works with nothing installed at all — every other
dependency is there to upgrade quality (a real graph, a real LLM, semantic
retrieval, real pretraining), not to make the thing run at all.

---

## Limitations, Stated Plainly

- All data is synthetic. No real operator network data was used anywhere
  in this repository.
- The signal-native foundation model proof of concept is small-scale by
  necessity (no GPU access) — see Part 3 above for exact numbers and what
  would be needed to scale it up.
- The KPI translator's offline fallback (no LLM configured) is
  intentionally weak; real numbers require a Groq key.
- This is solo, independent research, not an Ericsson product or
  affiliated with any specific employer initiative referenced in the
  background research for Part 3.

## License

MIT. Use, fork, adapt as you like.
