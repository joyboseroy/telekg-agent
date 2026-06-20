"""
TeleKG Multi-Agent System

Five specialized agents connected via LangGraph state machine:

  1. NetworkUnderstandingAgent  — monitors change events, updates KG
  2. ImpactAnalysisAgent        — traverses KG to find what a change affects
  3. RootCauseAgent             — backward-chains through KPI causal graph
  4. TestGenerationAgent        — generates test code from KG traversal results
  5. VerificationAgent          — (stub) runs tests, writes results back to KG

The LLM (Groq, e.g. llama-3.1-8b-instant) provides:
  - Natural language summaries of graph traversal results
  - Test code generation given a feature + KPI + context
  - Root cause narratives

The KG provides:
  - All structural knowledge (causal links, feature coverage, cell topology)
  - The LLM is NEVER asked to recall telecom facts from training data

This separation is the key architectural contribution:
  Graph = system knowledge | LLM = language + code generation

Set GROQ_API_KEY in your environment (or pass api_key= directly) to use a
real LLM. With no key configured, the pipeline runs end-to-end against an
offline rule-based stub instead — useful for quick smoke tests, but the
narration/test-code quality will be much lower than with a real model.
"""

from __future__ import annotations
import os
import json
from typing import TypedDict, Annotated
from telekg.reasoner import TeleKGReasoner

# ── LangGraph (optional — sequential fallback runner works without it) ──
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── Groq SDK (optional — offline stub used if not installed/configured) ──
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ── Shared State ──────────────────────────────────────────────────────

class KnowledgePlaneState(TypedDict):
    """State flowing through all agents in the pipeline."""
    # Input
    event: dict                        # The change event that triggered the pipeline

    # Populated by ImpactAnalysisAgent
    impact: dict                       # release_impact() result

    # Populated by RootCauseAgent
    root_cause_summary: str            # Natural language RCA narrative

    # Populated by TestGenerationAgent
    generated_tests: list[dict]        # [{name, code, feature, priority}]

    # Populated by VerificationAgent
    test_results: list[dict]           # [{name, status, output}]

    # Audit trail
    agent_log: list[str]


# ── LLM Wrapper (Groq) ───────────────────────────────────────────────────

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


class LLMWrapper:
    """
    Thin wrapper around the Groq chat completions API.

    Pass offline=True to force the rule-based stub (no network calls,
    no API key needed) — useful for CI or quick structural smoke tests.
    Otherwise, looks for GROQ_API_KEY in the environment; if missing or
    invalid, degrades to the same stub with a printed warning rather than
    crashing the pipeline.
    """

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, api_key: str = None, offline: bool = False):
        self.model = model
        self.offline = offline or not GROQ_AVAILABLE
        self.client = None
        if not self.offline:
            key = api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                print("[WARN] No GROQ_API_KEY found in environment and no api_key passed. "
                      "Falling back to offline stub. Get a free key at https://console.groq.com")
                self.offline = True
            else:
                self.client = Groq(api_key=key)

    def generate(self, prompt: str) -> str:
        if self.offline:
            return self._stub_response(prompt)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[WARN] Groq call failed ({e.__class__.__name__}: {e}); "
                  f"falling back to offline stub for this call.")
            return self._stub_response(prompt)

    def _stub_response(self, prompt: str) -> str:
        """Rule-based stub used when no Groq API key is configured."""
        if "test" in prompt.lower() and "def " in prompt.lower():
            return "[STUB] Test code would be generated here by the LLM."
        if "root cause" in prompt.lower():
            return "[STUB] Root cause analysis narrative would be generated here."
        if "summarize" in prompt.lower() or "summary" in prompt.lower():
            return "[STUB] Impact summary would be generated here."
        return "[STUB] LLM response."


# ── Agent Implementations ─────────────────────────────────────────────

class NetworkUnderstandingAgent:
    """
    Receives a ChangeEvent and updates the KG.
    Also classifies the event to determine which downstream agents to activate.
    """
    def __init__(self, reasoner: TeleKGReasoner, llm: LLMWrapper):
        self.reasoner = reasoner
        self.llm = llm

    def run(self, state: KnowledgePlaneState) -> KnowledgePlaneState:
        event = state["event"]
        log = state.get("agent_log", [])
        log.append(f"[NetworkUnderstandingAgent] Processing event: {event['event_id']} "
                   f"({event['event_type']}) — {event['description']}")
        state["agent_log"] = log
        return state


class ImpactAnalysisAgent:
    """
    Traverses the KG to determine what a change event could affect:
      - features touched by the release
      - KPIs at risk (direct + downstream via causal graph)
      - cells at risk
      - required test cases
    """
    def __init__(self, reasoner: TeleKGReasoner, llm: LLMWrapper):
        self.reasoner = reasoner
        self.llm = llm

    def run(self, state: KnowledgePlaneState) -> KnowledgePlaneState:
        event = state["event"]
        log = state.get("agent_log", [])

        if event["event_type"] == "SW_UPGRADE":
            release_id = event["trigger"]
            impact = self.reasoner.release_impact(release_id)
        else:
            # For config changes, outages, etc. — use affected cells
            impact = {
                "release": event["trigger"],
                "features": [],
                "direct_kpis": ["Accessibility", "Retainability"],
                "downstream_kpis": ["User_Experience"],
                "at_risk_cells": event.get("affected_cells", []),
                "required_tests": [],
            }

        # LLM summarises the graph traversal result in human language
        prompt = f"""
You are a telecom network engineer. A change event just occurred:
Event: {event['description']}
Event type: {event['event_type']}

The Knowledge Graph analysis shows:
- Features affected: {', '.join(impact.get('features', []))}
- KPIs directly at risk: {', '.join(impact.get('direct_kpis', []))}
- KPIs downstream (via causal propagation): {', '.join(impact.get('downstream_kpis', []))}
- Cells at risk: {len(impact.get('at_risk_cells', []))} cells

In 2-3 sentences, summarise the risk to the network and which KPIs need immediate attention.
"""
        summary = self.llm.generate(prompt)

        impact["summary"] = summary
        state["impact"] = impact
        log.append(f"[ImpactAnalysisAgent] Impact: {len(impact.get('features',[]))} features, "
                   f"{len(impact.get('direct_kpis',[]))} direct KPIs, "
                   f"{len(impact.get('at_risk_cells',[]))} cells at risk")
        state["agent_log"] = log
        return state


class RootCauseAgent:
    """
    Given degraded KPIs in the impact analysis, trace backward through
    the KPI causal graph to identify root causes.
    Used for post-event root cause analysis (RCA).
    """
    def __init__(self, reasoner: TeleKGReasoner, llm: LLMWrapper):
        self.reasoner = reasoner
        self.llm = llm

    def run(self, state: KnowledgePlaneState) -> KnowledgePlaneState:
        impact = state.get("impact", {})
        log = state.get("agent_log", [])

        rca_results = {}
        for kpi in impact.get("direct_kpis", [])[:3]:  # top 3 for brevity
            chain = self.reasoner.root_cause_chain(kpi)
            rca_results[kpi] = chain

        prompt = f"""
You are a telecom RCA specialist. The following KPIs are degraded:
{', '.join(impact.get('direct_kpis', []))}

Causal chain analysis from the Knowledge Graph:
{json.dumps(rca_results, indent=2)}

Write a concise root cause narrative (3-4 sentences) explaining the causal chain
that led to these KPI degradations, and the most likely root cause to investigate first.
"""
        narrative = self.llm.generate(prompt)
        state["root_cause_summary"] = narrative
        log.append(f"[RootCauseAgent] RCA complete for {len(rca_results)} KPIs")
        state["agent_log"] = log
        return state


class TestGenerationAgent:
    """
    The core contribution of the paper:
    Uses the KG traversal (required_tests from ImpactAnalysisAgent)
    to generate concrete Python test code via the LLM.

    The KG tells WHAT to test (feature/KPI/priority).
    The LLM generates HOW to test it (actual pytest code).
    """
    def __init__(self, reasoner: TeleKGReasoner, llm: LLMWrapper):
        self.reasoner = reasoner
        self.llm = llm

    def _generate_test_code(self, test: dict, impact: dict) -> str:
        """Generate pytest code for a single test case."""
        prompt = f"""
You are a 5G RAN test automation engineer. Generate a pytest test function.

Test case: {test['name']}
Priority: {test['priority']}
Feature under test: the feature that relates to this test
KPIs being validated: These are telecom KPIs measuring network health.

The test should:
1. Have a docstring explaining what it validates
2. Call a hypothetical network_client.get_kpi(cell_id, kpi_name) to get KPI values
3. Assert that KPI values are within acceptable thresholds
4. Use pytest.mark.{test['priority'].lower()} decorator
5. Be realistic but use mock/fixture data

Generate ONLY the Python function, no imports needed:
def {test['name']}(network_client, cell_fixture):
"""
        code = self.llm.generate(prompt)
        # If offline/stub, generate a template
        if "[STUB]" in code or not code.strip().startswith("def "):
            code = self._template_test(test)
        return code

    def _template_test(self, test: dict) -> str:
        """Fallback template when LLM unavailable."""
        return f'''
import pytest

@pytest.mark.{test["priority"].lower()}
def {test["name"]}(network_client, cell_fixture):
    """
    Auto-generated by TeleKG-Agent.
    Validates KPIs affected by feature change.
    Test ID: {test["id"]}
    """
    for cell_id in cell_fixture.at_risk_cells:
        kpi_values = network_client.get_kpi_snapshot(cell_id)
        # KPI threshold assertions (thresholds from KG)
        assert kpi_values.get("Accessibility", 100) >= 95.0, (
            f"Accessibility below threshold on {{cell_id}}"
        )
        assert kpi_values.get("Latency", 0) <= 20.0, (
            f"Latency above threshold on {{cell_id}}"
        )
'''.strip()

    def run(self, state: KnowledgePlaneState) -> KnowledgePlaneState:
        impact = state.get("impact", {})
        log = state.get("agent_log", [])
        event = state["event"]

        # Get prioritised test list from reasoner
        if event["event_type"] == "SW_UPGRADE":
            tests = self.reasoner.prioritised_tests_for_release(event["trigger"])
        else:
            tests = impact.get("required_tests", [])

        generated = []
        for test in tests:
            code = self._generate_test_code(test, impact)
            generated.append({
                "id": test["id"],
                "name": test["name"],
                "priority": test["priority"],
                "code": code,
                "source": "TeleKG-Agent",
                "trigger_event": event["event_id"],
            })

        state["generated_tests"] = generated
        log.append(f"[TestGenerationAgent] Generated {len(generated)} test cases "
                   f"(priorities: {[t['priority'] for t in generated]})")
        state["agent_log"] = log
        return state


class VerificationAgent:
    """
    Stub: would run the generated tests against a test environment
    and write pass/fail results back into the KG as TestResult nodes.
    """
    def __init__(self, reasoner: TeleKGReasoner, llm: LLMWrapper):
        self.reasoner = reasoner
        self.llm = llm

    def run(self, state: KnowledgePlaneState) -> KnowledgePlaneState:
        log = state.get("agent_log", [])
        n_tests = len(state.get("generated_tests", []))
        log.append(f"[VerificationAgent] {n_tests} tests queued for execution "
                   f"(execution against live network not implemented in prototype)")
        state["test_results"] = []
        state["agent_log"] = log
        return state


# ── LangGraph Pipeline ─────────────────────────────────────────────────

def build_pipeline(reasoner: TeleKGReasoner, llm: LLMWrapper):
    """
    Constructs the LangGraph state machine.
    Flow: NetworkUnderstanding → ImpactAnalysis → RootCause → TestGeneration → Verification
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    net_agent  = NetworkUnderstandingAgent(reasoner, llm)
    imp_agent  = ImpactAnalysisAgent(reasoner, llm)
    rca_agent  = RootCauseAgent(reasoner, llm)
    test_agent = TestGenerationAgent(reasoner, llm)
    ver_agent  = VerificationAgent(reasoner, llm)

    graph = StateGraph(KnowledgePlaneState)
    graph.add_node("network_understanding", net_agent.run)
    graph.add_node("impact_analysis",       imp_agent.run)
    graph.add_node("root_cause",            rca_agent.run)
    graph.add_node("test_generation",       test_agent.run)
    graph.add_node("verification",          ver_agent.run)

    graph.set_entry_point("network_understanding")
    graph.add_edge("network_understanding", "impact_analysis")
    graph.add_edge("impact_analysis",       "root_cause")
    graph.add_edge("root_cause",            "test_generation")
    graph.add_edge("test_generation",       "verification")
    graph.add_edge("verification",          END)

    return graph.compile()


# ── Fallback Sequential Runner (no LangGraph) ─────────────────────────

def run_pipeline_sequential(event: dict, reasoner: TeleKGReasoner, llm: LLMWrapper) -> dict:
    """Runs the same agent sequence without LangGraph dependency."""
    state: KnowledgePlaneState = {
        "event": event,
        "impact": {},
        "root_cause_summary": "",
        "generated_tests": [],
        "test_results": [],
        "agent_log": [],
    }

    agents = [
        NetworkUnderstandingAgent(reasoner, llm),
        ImpactAnalysisAgent(reasoner, llm),
        RootCauseAgent(reasoner, llm),
        TestGenerationAgent(reasoner, llm),
        VerificationAgent(reasoner, llm),
    ]

    for agent in agents:
        state = agent.run(state)

    return state
