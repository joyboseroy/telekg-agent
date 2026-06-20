"""
KPI Translator: Natural-Language KPI Definition -> PM Counter Formula -> Value

This is the system that addresses: "dynamically translate user-defined
(custom) KPI definitions, map them to available PM counters, retrieve the
required time-series data, and compute the KPI accurately and efficiently
in near real-time or on-demand."

Pipeline:
  1. User types a free-text KPI definition, e.g.
     "percentage of dropped calls" or "average time for a connection to be
     set up successfully"
  2. RETRIEVAL: embed the query, retrieve top-k candidate PM counters from
     the counter dictionary (telekg/pm_registry.py) by semantic similarity,
     using counter descriptions as the retrieval corpus (this is the RAG step
     — grounding the LLM in the actual available counters rather than letting
     it invent counter names).
  3. GENERATION: the LLM is given the user's definition + retrieved candidate
     counters, and asked to either (a) match to an existing registered KPI
     formula, or (b) compose a NEW formula from the retrieved counters if no
     exact match exists.
  4. VALIDATION: the generated formula is checked — every counter it
     references must exist in the registry (no hallucinated counters
     allowed) and the formula must be evaluable.
  5. COMPUTATION: the validated formula is evaluated against live/synthetic
     counter values for the requested cell(s) and time window.

The retrieval step is what makes this RAG rather than a bare LLM call: the
LLM never free-associates a counter name, it can only select from counters
the retriever surfaced as candidates, which are always real entries in
PM_COUNTERS.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass

from telekg.pm_registry import PM_COUNTERS, KPI_REGISTRY, INFORMAL_ALIASES, compute_kpi

# KPITranslator accepts any object with a .generate(prompt) -> str method.
# agents.pipeline.LLMWrapper (Groq-backed, with offline stub fallback) is
# the expected implementation — see run_kpi_eval.py for wiring.


# ─────────────────────────────────────────────
# Embedding / retrieval backend
# ─────────────────────────────────────────────

class CounterRetriever:
    """
    Retrieves the top-k most semantically relevant PM counters for a
    free-text query, using sentence embeddings over counter descriptions.
    Falls back to TF-IDF-style keyword overlap if no embedding model
    is available (keeps the pipeline runnable offline).
    """

    def __init__(self, use_embeddings: bool = True):
        self.corpus_ids = list(PM_COUNTERS.keys())
        self.corpus_texts = [
            f"{c.name}: {c.description} (unit: {c.unit})"
            for c in PM_COUNTERS.values()
        ]
        self.use_embeddings = use_embeddings
        self._model = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._corpus_emb = self._model.encode(self.corpus_texts, normalize_embeddings=True)
            except Exception:
                self.use_embeddings = False

    def retrieve(self, query: str, top_k: int = 8) -> list[str]:
        if self.use_embeddings and self._model is not None:
            import numpy as np
            q_emb = self._model.encode([query], normalize_embeddings=True)[0]
            sims = self._corpus_emb @ q_emb
            ranked = sorted(zip(self.corpus_ids, sims), key=lambda x: -x[1])
            return [cid for cid, _ in ranked[:top_k]]
        return self._keyword_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int) -> list[str]:
        """Fallback: token-overlap scoring against counter name + description."""
        q_tokens = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for cid, text in zip(self.corpus_ids, self.corpus_texts):
            t_tokens = set(re.findall(r"[a-z]+", text.lower()))
            overlap = len(q_tokens & t_tokens)
            scored.append((cid, overlap))
        scored.sort(key=lambda x: -x[1])
        # If literally nothing overlapped, return a broad default set
        if scored[0][1] == 0:
            return self.corpus_ids[:top_k]
        return [cid for cid, score in scored[:top_k] if score > 0] or self.corpus_ids[:top_k]


# ─────────────────────────────────────────────
# LLM formula generator
# ─────────────────────────────────────────────

@dataclass
class TranslationResult:
    user_query: str
    matched_existing_kpi: str | None     # name in KPI_REGISTRY, if matched
    generated_formula: str | None        # formula string if newly composed
    counters_used: list[str]
    valid: bool
    validation_errors: list[str]
    computed_value: float | None = None
    raw_llm_output: str = ""


class KPITranslator:
    """
    Orchestrates retrieval + LLM generation + validation + computation.
    Works in offline mode (rule-based) when no LLM is configured, so the
    whole pipeline and its evaluation harness can run without API keys.
    """

    def __init__(self, llm=None, retriever: CounterRetriever = None):
        self.llm = llm
        self.retriever = retriever or CounterRetriever()

    # ── Step 1+2: retrieval-augmented prompt construction ──────────────

    def _build_prompt(self, user_query: str, candidate_counters: list[str]) -> str:
        counter_block = "\n".join(
            f"- {cid}: {PM_COUNTERS[cid].description} (unit: {PM_COUNTERS[cid].unit})"
            for cid in candidate_counters
        )
        # Only surface existing KPIs whose formulas use at least one of the
        # RETRIEVED candidate counters. Previously this listed all 10
        # registered KPIs on every call regardless of what retrieval
        # surfaced, which let the LLM match against KPIs that had nothing
        # to do with the retrieved (and therefore actually relevant)
        # counters -- undermining the whole point of grounding via RAG.
        candidate_set = set(candidate_counters)
        relevant_kpis = {
            k: v for k, v in KPI_REGISTRY.items()
            if candidate_set & set(v.counters_used)
        }
if relevant_kpis:
            existing_kpis = "\n".join(f"- {k}: {v.formula_str}" for k, v in relevant_kpis.items())
            existing_kpis_block = f"""Existing registered KPI formulas that use one or more of the
retrieved counters above:
{existing_kpis}

Decide whether the user's definition asks for the SAME underlying metric as
one of these existing KPIs, or something DIFFERENT that merely uses an
overlapping counter.

- MATCH (set matched_existing_kpi, leave formula null) only if the
  underlying quantity being measured is the same, even if phrased
  differently. Example: "what fraction of connections succeed" IS
  Accessibility -- same quantity, different words. Match it.
- DO NOT MATCH (compose a new formula instead) if the user is asking about
  a different scope, direction, or aggregation than what the existing KPI
  computes, even though it touches related counters. Example: "average
  connection duration in seconds" is NOT Cell_Availability, even though
  both touch cell-time-related counters -- they measure different things.
  Another example: "uplink packet loss" is NOT the registered Packet_Loss
  KPI if that KPI is defined on downlink counters -- uplink and downlink
  are different metrics, compose a separate uplink formula instead.
- If genuinely uncertain whether it's the same metric, prefer composing a
  new formula over forcing an imprecise match -- an explicit new formula is
  more useful than mislabelling against the wrong existing KPI.
- Example: "average round trip delay" IS the registered Latency KPI
  (RTT_SUM_MS / RTT_SAMPLE_COUNT) -- match it, do not compose a different
  denominator like cell uptime."""
        else:
            existing_kpis_block = (
                "No existing registered KPI uses any of the retrieved counters above. "
                "You must compose a new formula from the retrieved counters, or return "
                "null for both matched_existing_kpi and formula if none of the retrieved "
                "counters are actually relevant to the user's definition."
            )

        return f"""You are a telecom KPI translation assistant. A user has typed a
custom KPI definition in plain language. Your job is to map it to a formula
using ONLY the PM counters listed below. Never invent a counter name that is
not in this list.

User's KPI definition: "{user_query}"

Available PM counters (retrieved as relevant to this query):
{counter_block}

{existing_kpis_block}

Respond ONLY in this JSON format, no other text:
{{
  "matched_existing_kpi": "<KPI name from the existing list above, or null>",
  "formula": "<arithmetic formula string using ONLY counter names listed above, or null if matched_existing_kpi is set>",
  "counters_used": ["<counter1>", "<counter2>", ...],
  "reasoning": "<one sentence>"
}}
"""

    # ── Step 3: LLM call (or offline rule-based fallback) ───────────────

    def _call_llm(self, prompt: str, user_query: str, candidates: list[str]) -> str:
        # Detect a non-functional LLM (either None, or agents.pipeline.LLMWrapper
        # running in its own offline/stub mode) and route to OUR offline fallback
        # instead of letting the wrapper's generic "[STUB] LLM response." string
        # get parsed as if it were real model output.
        is_stub_wrapper = getattr(self.llm, "offline", False)
        if self.llm is None or is_stub_wrapper:
            return self._offline_translate(user_query, candidates)
        try:
            output = self.llm.generate(prompt)
            if not output or output.strip().startswith("[STUB]"):
                return self._offline_translate(user_query, candidates)
            return output
        except Exception:
            return self._offline_translate(user_query, candidates)

    def _offline_translate(self, user_query: str, candidates: list[str]) -> str:
        """
        Rule-based fallback used when no LLM is configured. Checks the
        informal alias table first (paraphrase matching), then falls back
        to a naive ratio-formula guess from the top-2 retrieved counters.
        This exists purely so the pipeline is runnable end-to-end offline;
        the paper's reported accuracy numbers come from real LLM runs.
        """
        q_lower = user_query.lower().strip()
        for alias, kpi_name in INFORMAL_ALIASES.items():
            if alias in q_lower:
                return json.dumps({
                    "matched_existing_kpi": kpi_name,
                    "formula": None,
                    "counters_used": KPI_REGISTRY[kpi_name].counters_used,
                    "reasoning": f"Offline alias match: '{alias}' -> {kpi_name}"
                })
        # No alias hit: naive guess, compose a ratio of the top 2 candidates
        if len(candidates) >= 2:
            a, b = candidates[0], candidates[1]
            return json.dumps({
                "matched_existing_kpi": None,
                "formula": f"{a} / {b} * 100",
                "counters_used": [a, b],
                "reasoning": "Offline fallback: naive ratio of top-2 retrieved counters (low confidence)"
            })
        return json.dumps({
            "matched_existing_kpi": None, "formula": None,
            "counters_used": [], "reasoning": "No candidates retrieved"
        })

    # ── Step 4: validation ───────────────────────────────────────────────

    def _validate(self, parsed: dict) -> tuple[bool, list[str]]:
        errors = []
        matched = parsed.get("matched_existing_kpi")
        formula = parsed.get("formula")
        counters_used = parsed.get("counters_used", [])

        if matched and matched not in KPI_REGISTRY:
            errors.append(f"matched_existing_kpi '{matched}' is not a registered KPI")

        if not matched and not formula:
            errors.append("Neither matched_existing_kpi nor formula provided")

        for c in counters_used:
            if c not in PM_COUNTERS:
                errors.append(f"Hallucinated counter not in registry: '{c}'")

        if formula:
            # Every counter token referenced in the formula string must be valid
            tokens = re.findall(r"[A-Z][A-Z0-9_]+", formula)
            for tok in tokens:
                if tok not in PM_COUNTERS:
                    errors.append(f"Formula references unknown counter: '{tok}'")

        return (len(errors) == 0), errors

    # ── Step 5: computation ──────────────────────────────────────────────

    def _compute(self, parsed: dict, counter_values: dict[str, float]) -> float | None:
        matched = parsed.get("matched_existing_kpi")
        if matched and matched in KPI_REGISTRY:
            try:
                return compute_kpi(matched, counter_values)
            except Exception:
                return None

        formula = parsed.get("formula")
        if not formula:
            return None
        try:
            # Safe-ish eval: only allow counter tokens + numbers + arithmetic ops
            tokens = re.findall(r"[A-Z][A-Z0-9_]+", formula)
            safe_formula = formula
            local_ns = {}
            for tok in tokens:
                if tok in counter_values:
                    local_ns[tok] = counter_values[tok]
                else:
                    return None
            allowed_chars = set("0123456789.+-*/() ")
            stripped = formula
            for tok in tokens:
                stripped = stripped.replace(tok, "")
            if not set(stripped.strip()) <= allowed_chars:
                return None  # formula contains disallowed characters -> reject
            return eval(formula, {"__builtins__": {}}, local_ns)
        except Exception:
            return None

    # ── Public API ────────────────────────────────────────────────────────

    def translate(self, user_query: str, counter_values: dict[str, float] = None,
                   top_k: int = 8) -> TranslationResult:
        candidates = self.retriever.retrieve(user_query, top_k=top_k)
        prompt = self._build_prompt(user_query, candidates)
        raw_output = self._call_llm(prompt, user_query, candidates)

        try:
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```json?|```$", "", cleaned, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
        except Exception:
            return TranslationResult(
                user_query=user_query, matched_existing_kpi=None,
                generated_formula=None, counters_used=[],
                valid=False, validation_errors=["LLM output was not valid JSON"],
                raw_llm_output=raw_output,
            )

        valid, errors = self._validate(parsed)
        computed = None
        if valid and counter_values:
            computed = self._compute(parsed, counter_values)

        return TranslationResult(
            user_query=user_query,
            matched_existing_kpi=parsed.get("matched_existing_kpi"),
            generated_formula=parsed.get("formula"),
            counters_used=parsed.get("counters_used", []),
            valid=valid,
            validation_errors=errors,
            computed_value=computed,
            raw_llm_output=raw_output,
        )
