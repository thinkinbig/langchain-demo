# Project Status: Deep Research V2

## 🚀 Current Phase: Deep Research Upgrade
**Goal:** Upgrade the MVP "General Researcher" to a "Deep Research Agent" capable of multi-hop causal reasoning, strict fact verification, and deep navigation.

---

## 📋 Phase 2 Checklist (Deep Research)

### 1. Causal Chain Navigation (Gap 1)
- [ ] **Dependency Graph Support**: Modify `LeadResearcher` to generate tasks with dependencies (Task B waits for Task A output).
- [ ] **Context Injection**: Pass output from Step A (e.g., "Company Name") into Step B's prompt dynamically.
- [ ] **Hard Gating Logic**: Prevent subagents from running if prerequisite data is missing/null.

### 2. Verification & Hallucination Control (Gap 2 - FACT)
- [ ] **Verifier Node**: Add a new graph node dedicated to checking claims against sources.
- [ ] **Claim Extraction**: Implement regex/LLM logic to parse claims from citations.
- [ ] **Entailment Check**: Implement "Support/Refute/Irrelevant" check against source text.

### 3. Deep Navigation & Tools (Gap 3)
- [ ] **Browse Tool**: Implement `browse_page` (using Jina/Firecrawl or simple requests) to read full page content, not just snippets.
- [ ] **Recursive Search**: Allow subagents to click links found in page content (Depth > 1).
- [ ] **File Parsing**: Capability to read PDF/CSV files from URLs (optional for V2.0).

### 4. Automated Evaluation (Gap 4 - RACE)
- [ ] **Evaluation Script**: Port `deep_research_evaluation_framework.md` logic to python script.
- [ ] **Dataset**: Create/Import 10 "Golden" Deep Research questions (from GAIA/DeepSearchQA).
- [ ] **Judge Implementation**: Use LLM-as-a-Judge to score reports on Comprehensiveness and Factuality.

---

## 📜 Archive: MVP Completion (Phase 1)
**Status:** ✅ Completed (Dec 2025)

### Core Functions
- ✅ **LeadResearcher Node**: Plan creation & refinement.
- ✅ **Subagent Nodes**: Parallel Tavily search.
- ✅ **Synthesis**: Multi-source aggregation.
- ✅ **Cost Control**: `QueryBudget` and `DailyBudget` enforced.

### Reliability
- ✅ **Retry Logic**: Implemented for Search and Verification.
- ✅ **Tests**: Unit (nodes), Integration (graph), and E2E tests passing.

### Documentation
- ✅ **MVP Analysis**: Requirements defined.
- ✅ **Metrics**: Quantified KPIs established.
