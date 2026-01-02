"""Multi-agent research system graph"""

from typing import Dict, List, Literal

import context_manager
import tools
from context_formatter import ContextFormatter
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from memory_helpers import (
    content_metadata_to_string,
    create_content_metadata,
    extract_evidence_summaries,
    extract_findings_metadata,
)
from retrieval import RetrievalResult, RetrievalService
from schemas import (
    Citation,
    CitationAgentState,
    DecisionState,
    Finding,
    LeadResearcherState,
    ResearchState,
    ResearchTasks,
    SubagentState,
    SynthesisResult,
    SynthesizerState,
)

# LLM Configuration - Lazy Loading Pattern
# LLMs are initialized on first use, not at module import time
# This allows tests to mock LLMs without needing API keys

_lead_llm = None
_subagent_llm = None

# Import tracing for local logging (optional)
try:
    from tracing import get_callbacks
except ImportError:
    def get_callbacks():
        return []


def get_lead_llm():
    """Get or create the lead LLM instance (lazy loading)"""
    global _lead_llm
    if _lead_llm is None:
        _lead_llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.3,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _lead_llm


def get_subagent_llm():
    """Get or create the subagent LLM instance (lazy loading)"""
    global _subagent_llm
    if _subagent_llm is None:
        _subagent_llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
            temperature=0.3,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _subagent_llm


def process_structured_response(response, state, fallback_func=None):
    """
    Standardized validation and state update for retry loop.

    Args:
        response: Output from .with_structured_output(..., include_raw=True)
        state: Current state dict
        fallback_func: Optional callable to generate fallback state
            if max retries reached

    Returns:
        dict: State update
    """
    retry_count = state.get("retry_count", 0)
    parsing_error = response.get("parsing_error")

    if parsing_error:
        print(f"  ❌ Validation failed: {parsing_error}")

        # Check max retries (e.g., 3 attempts total: 0, 1, 2)
        if retry_count >= 2:
            print(f"  ❌ Max retries ({retry_count + 1}) reached")
            if fallback_func:
                print("  ⚠️  Using fallback logic")
                fallback_update = fallback_func(state)
                # Ensure fallback clears the error state
                return {**fallback_update, "error": None, "retry_count": 0}

            # If no fallback, just propagate the error or decide to end
            return {"error": str(parsing_error), "retry_count": retry_count + 1}

        # Route back for retry
        return {
            "error": str(parsing_error),
            "retry_count": retry_count + 1
        }

    # Success
    # Signal to caller that retrieval was successful, caller handles "parsed"
    return None


def should_retry(state: ResearchState) -> bool:
    """Common condition to check if we should loop back"""
    return bool(state.get("error"))


def lead_researcher_node(state: LeadResearcherState):
    """LeadResearcher: Analyze query, create plan, generate subagent tasks"""
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")

    print(f"\n🔍 [LeadResearcher] Analyzing query (iteration {iteration_count + 1})...")
    print(f"   Query: {query[:80]}...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    # Optimized prompt for query analysis and task generation
    from prompts import (
        LEAD_RESEARCHER_INITIAL,
        LEAD_RESEARCHER_REFINE,
        LEAD_RESEARCHER_RETRY,
        LEAD_RESEARCHER_SYSTEM,
    )

    if iteration_count == 0:
        prompt_content = LEAD_RESEARCHER_INITIAL.format(
            query=query,
            scratchpad=state.get("scratchpad", "")
        )
    else:
        existing_findings = state.get("subagent_findings", [])
        findings_summary = "\n".join([
            f"- {f.get('task', 'Unknown')[:50]}: {f.get('summary', '')[:80]}"
            for f in existing_findings[:3]
        ])
        prompt_content = LEAD_RESEARCHER_REFINE.format(
            query=query,
            findings_summary=findings_summary,
            scratchpad=state.get("scratchpad", "")
        )

    # Add feedback if retrying - Standard Pattern
    if last_error:
        prompt_content = LEAD_RESEARCHER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    # Invoke LLM
    structured_llm = get_lead_llm().with_structured_output(
        ResearchTasks, include_raw=True
    )

    # RAG: Retrieve context dynamically (Limited to k=2 for planner efficiency)
    print(f"  🧠 [RAG] Retrieving context for: {query[:50]}...")
    retrieved_context, _ = context_manager.retrieve_knowledge(query, k=2)

    # Inject retrieved context into prompt
    rag_instructions = (
        f"\n\n<internal_knowledge>\n{retrieved_context}\n</internal_knowledge>\n"
        "Note: Use the above internal knowledge if relevant to the task breakdown."
    )

    # Invoke LLM without static knowledge base in system prompt
    system_prompt = context_manager.get_system_context(LEAD_RESEARCHER_SYSTEM, include_knowledge=False)

    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content + rag_instructions)
    ])

    # Use helper to process retry logic
    def fallback(s):
        from schemas import ResearchTask
        return {
            "research_plan": f"Research plan for: {s['query']} (Fallback)",
            "subagent_tasks": [
                ResearchTask(
                    id="task_1",
                    description=f"Research: {s['query']}",
                    rationale="Fallback"
                ),
                ResearchTask(
                    id="task_2",
                    description=f"Info: {s['query']}",
                    rationale="Fallback"
                )
            ],
            "iteration_count": s.get("iteration_count", 0) + 1,
            "scratchpad": s.get("scratchpad", "Fallback due to error")
        }

    retry_state = process_structured_response(response, state, fallback)
    if retry_state:
        # If retry_state is returned, it means we either failed (and are looping)
        # or we hit max retries and are returning the fallback.
        # In both cases, we return deeply.
        return retry_state

    # Success case
    parsed_result = response["parsed"]
    tasks = parsed_result.tasks

    plan = f"Research plan for: {query}\nTasks: {len(tasks)} sub-tasks"
    print(f"  ✅ Created {len(tasks)} sub-tasks")
    for i, task in enumerate(tasks, 1):
        print(f"     {i}. {task.description[:60]}... (ID: {task.id})")

    return {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "iteration_count": iteration_count + 1,
        "error": None,
        "retry_count": 0,
        "scratchpad": parsed_result.scratchpad
    }


# --- SUBAGENT SUBGRAPH COMPONENTS ---

def retrieve_node(state: SubagentState):
    """Node: Retrieve internal knowledge (RAG)"""
    tasks = state.get("subagent_tasks", [])
    if not tasks:
        return {}

    # Get task description
    rs_task = tasks[0]
    task_description = rs_task.description if hasattr(rs_task, "description") else str(rs_task)

    # Get visited sources from unified format
    visited_sources = state.get("visited_sources", [])
    visited_identifiers = [vs.identifier if hasattr(vs, "identifier") else vs.get("identifier", "")
                          for vs in visited_sources if (hasattr(vs, "source_type") and getattr(vs, "source_type", "") == "internal")
                          or (isinstance(vs, dict) and vs.get("source_type") == "internal")]

    print(f"  🧠 [RAG] Retrieving context for task: {task_description[:50]}...")

    # Use unified retrieval service
    result = RetrievalService.retrieve_internal(
        query=task_description,
        visited_sources=visited_identifiers,
        k=4
    )

    # Store in state using new unified format
    return {
        "internal_result": result,
        "task_description": task_description,  # Cache for downstream
    }


def web_search_node(state: SubagentState):
    """Node: Perform web search if RAG failed"""
    task_description = state.get("task_description", "")

    if not task_description:
        # Fallback if not cached
        tasks = state.get("subagent_tasks", [])
        rs_task = tasks[0]
        task_description = rs_task.description if hasattr(rs_task, "description") else str(rs_task)

    print(f"\n🔎 [Subagent-Web] Internal knowledge empty. Performing Web Search: {task_description[:60]}...")

    # Get visited URLs from unified format
    visited_sources = state.get("visited_sources", [])
    visited_urls = [vs.identifier if hasattr(vs, "identifier") else vs.get("identifier", "")
                    for vs in visited_sources if (hasattr(vs, "source_type") and getattr(vs, "source_type", "") == "web")
                    or (isinstance(vs, dict) and vs.get("source_type") == "web")]

    # Use unified retrieval service
    result = RetrievalService.retrieve_web(
        query=task_description,
        visited_urls=visited_urls,
        max_results=5,
        scrape_top_result=True
    )

    # Store in state using new unified format
    return {
        "web_result": result,
    }


def route_source_necessity(state: SubagentState) -> Literal["analysis_node", "web_search_node"]:
    """Gating Node: Check if internal knowledge is sufficient"""
    internal_result = state.get("internal_result")

    if internal_result:
        if isinstance(internal_result, RetrievalResult):
            if not internal_result.is_empty():
                print("  🚫 [Gating] Internal knowledge found. Skipping Web Search.")
                return "analysis_node"
        elif isinstance(internal_result, dict):
            # Handle dict representation (from state serialization)
            if internal_result.get("has_content") and not internal_result.get("content", "").startswith("(No relevant"):
                print("  🚫 [Gating] Internal knowledge found. Skipping Web Search.")
                return "analysis_node"

    print("  🔄 [Gating] Internal knowledge insufficient. Routing to Web Search.")
    return "web_search_node"


def analysis_node(state: SubagentState):
    """Node: Analyze gathered context (Internal or Web) and call tools"""
    from prompts import SUBAGENT_ANALYSIS, SUBAGENT_SYSTEM

    # Get task description
    task_description = state.get("task_description", "")
    if not task_description:
        tasks = state.get("subagent_tasks", [])
        if tasks:
            rs_task = tasks[0]
            task_description = rs_task.description if hasattr(rs_task, "description") else str(rs_task)

    # Get retrieval results (unified format)
    internal_result = state.get("internal_result")
    web_result = state.get("web_result")

    # Convert dict representation to RetrievalResult if needed (from state serialization)
    if internal_result and not isinstance(internal_result, RetrievalResult):
        if isinstance(internal_result, dict):
            from retrieval import RetrievalSource, Source
            sources = [Source(**s) for s in internal_result.get("sources", [])]
            internal_result = RetrievalResult(
                content=internal_result.get("content", ""),
                sources=sources,
                source_type=RetrievalSource.INTERNAL,
                has_content=internal_result.get("has_content", False)
            )
        else:
            internal_result = None

    if web_result and not isinstance(web_result, RetrievalResult):
        if isinstance(web_result, dict):
            from retrieval import RetrievalSource, Source
            sources = [Source(**s) for s in web_result.get("sources", [])]
            web_result = RetrievalResult(
                content=web_result.get("content", ""),
                sources=sources,
                source_type=RetrievalSource.WEB,
                has_content=web_result.get("has_content", False)
            )
        else:
            web_result = None

    # Get visited identifiers for context formatting (from unified format)
    visited_sources = state.get("visited_sources", [])
    visited_identifiers = []
    for vs in visited_sources:
        if hasattr(vs, "identifier"):
            visited_identifiers.append(vs.identifier)
        elif isinstance(vs, dict):
            visited_identifiers.append(vs.get("identifier", ""))

    # Use ContextFormatter to format context
    formatter = ContextFormatter()
    formatted_context, citation_instructions, all_sources = formatter.format_for_analysis(
        task=task_description,
        internal_result=internal_result,
        web_result=web_result,
        visited_identifiers=visited_identifiers
    )

    # Log context info and source tracking
    if internal_result and not internal_result.is_empty():
        source_ids = [s.identifier for s in internal_result.sources]
        print(f"     Found {len(internal_result.content)} chars of internal context from {len(internal_result.sources)} sources: {source_ids[:3]}")
    if web_result and not web_result.is_empty():
        source_urls = [s.identifier for s in web_result.sources]
        print(f"     Found {len(web_result.content)} chars of web context from {len(web_result.sources)} sources: {source_urls[:3]}")

    # Debug: Log total sources collected
    print(f"  📚 [Source Tracking] Total sources available: {len(all_sources)} ({[s.identifier[:30] for s in all_sources[:3]]})")

    # Create Analysis Prompt
    analysis_prompt_content = SUBAGENT_ANALYSIS.format(
        task=task_description,
        results=formatted_context + citation_instructions
    )

    # --- Agent Loop (Same as before) ---
    system_prompt = context_manager.get_system_context(SUBAGENT_SYSTEM, include_knowledge=True)

    # Define Tools
    from langchain_core.tools import tool

    @tool
    def submit_findings(summary: str, sources: List[Dict[str, str]] = None):
        """Submit the final synthesized findings."""
        return {"summary": summary, "sources": sources}

    tool_list = [tools.python_repl, tools.read_local_file, submit_findings]
    formatted_llm = get_subagent_llm().bind_tools(tool_list)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=analysis_prompt_content),
    ]

    for step in range(5):
        response = formatted_llm.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for t_call in response.tool_calls:
                t_name = t_call["name"]
                t_args = t_call["args"]
                t_id = t_call["id"]

                print(f"    🛠️ [CodeAgent] Call: {t_name}")

                if t_name == "python_repl":
                    try:
                        result = tools.python_repl(t_args.get("code", ""))
                        messages.append(ToolMessage(tool_call_id=t_id, content=str(result)))
                    except Exception as e:
                        messages.append(ToolMessage(tool_call_id=t_id, content=f"Error: {e}"))

                elif t_name == "read_local_file":
                    res = tools.read_local_file(t_args.get("file_path", ""))
                    messages.append(ToolMessage(tool_call_id=t_id, content=res))

                elif t_name == "submit_findings":
                    # Merging logic for sources
                    # Start with sources from retrieval results (ALWAYS include these)
                    final_sources = []
                    for source in all_sources:
                        if source.source_type.value == "internal":
                            final_sources.append({
                                "title": source.title or f"Internal Document: {source.identifier}",
                                "url": f"internal/{source.identifier}"
                            })
                        else:
                            final_sources.append({
                                "title": source.title or source.identifier,
                                "url": source.identifier
                            })

                    # Add tool-provided sources (from LLM) - merge with retrieval sources
                    tool_sources = t_args.get("sources")
                    if tool_sources:
                        for s in tool_sources:
                            # Check if source already exists by URL
                            url = s.get("url", "")
                            if url and not any(fs.get("url") == url for fs in final_sources):
                                final_sources.append(s)

                    # Log source collection for debugging
                    print(f"  📚 [Sources] Collected {len(final_sources)} sources: {[s.get('title', s.get('url', 'Unknown')) for s in final_sources[:3]]}")

                    # Store content metadata instead of full content (token optimization)
                    # Full content is available in formatted_context but we store only metadata
                    if formatted_context:
                        content_metadata = create_content_metadata(formatted_context, max_preview=200)
                        evidence_content = content_metadata_to_string(content_metadata)
                    else:
                        evidence_content = ""

                    finding = Finding(
                        task=task_description,
                        summary=t_args.get("summary", ""),
                        sources=final_sources,
                        content=evidence_content  # Now stores metadata dict as JSON string
                    )
                    print("  ✅ CodeAgent finished via tool.")
                    return {"subagent_findings": [finding]}
        else:
             print("  ⚠️  CodeAgent output text but no tool call. Re-prompting...")
             messages.append(HumanMessage(content="You generally must call `submit_findings`."))

    # Fail safe
    print("  ❌ CodeAgent failed to submit findings.")
    return {"subagent_findings": [Finding(task=task_description, summary="Analysis failed", sources=[])]}

# Define the Subgraph
subagent_workflow = StateGraph(SubagentState)
subagent_workflow.add_node("retrieve_node", retrieve_node)
subagent_workflow.add_node("web_search_node", web_search_node)
subagent_workflow.add_node("analysis_node", analysis_node)

subagent_workflow.add_edge(START, "retrieve_node")
subagent_workflow.add_conditional_edges(
    "retrieve_node",
    route_source_necessity,
    {
        "analysis_node": "analysis_node",
        "web_search_node": "web_search_node"
    }
)
subagent_workflow.add_edge("web_search_node", "analysis_node")
subagent_workflow.add_edge("analysis_node", END)

# Compile Subgraph
subagent_app = subagent_workflow.compile()



def subagent_node(state: SubagentState):
    """Wrapper to invoke the subagent subgraph and filter output"""
    # Invoke the subgraph
    # We pass the state directly. The subgraph runs and returns its final state.
    result_state = subagent_app.invoke(state)

    # Extract visited sources from retrieval results and convert to VisitedSource format
    from schemas import VisitedSource

    visited_sources = []

    # Get internal result sources
    internal_result = result_state.get("internal_result")
    if internal_result:
        if isinstance(internal_result, RetrievalResult):
            for source in internal_result.sources:
                visited_sources.append(VisitedSource(
                    identifier=source.identifier,
                    source_type="internal"
                ))
        elif isinstance(internal_result, dict):
            for s in internal_result.get("sources", []):
                visited_sources.append(VisitedSource(
                    identifier=s.get("identifier", ""),
                    source_type="internal"
                ))

    # Get web result sources
    web_result = result_state.get("web_result")
    if web_result:
        if isinstance(web_result, RetrievalResult):
            for source in web_result.sources:
                visited_sources.append(VisitedSource(
                    identifier=source.identifier,
                    source_type="web"
                ))
        elif isinstance(web_result, dict):
            for s in web_result.get("sources", []):
                visited_sources.append(VisitedSource(
                    identifier=s.get("identifier", ""),
                    source_type="web"
                ))

    # Filter output: return ONLY what needs to be merged to the parent state
    # This avoids 'InvalidConcurrentGraphUpdate' on non-reducer keys like 'subagent_tasks'
    return {
        "subagent_findings": result_state.get("subagent_findings", []),
        "visited_sources": visited_sources
    }


def filter_findings_node(state: ResearchState):
    """
    Guardrail Node: Sanitize and filter findings before synthesis.
    - Removes empty/failed findings.
    - Removes findings with no relevant content.
    - Deduplicates sources.
    """
    findings = state.get("subagent_findings", [])
    print(f"\n🛡️  [Guardrail] Filtering {len(findings)} findings...")

    valid_findings = []
    seen_content_hashes = set()

    for f in findings:
        # 1. Check for failure patterns in summary
        summary_lower = f.summary.lower()
        fail_patterns = [
            "no information found",
            "no search results",
            "unable to find",
            "failed to read",
            "analysis failed"
        ]
        if any(p in summary_lower for p in fail_patterns):
            print(f"     Dropped noise: {f.task[:40]}...")
            continue

        # 2. Check content length (heuristic relevance)
        if len(f.content) < 50 and len(f.summary) < 20:
            print(f"     Dropped empty/short: {f.task[:40]}...")
            continue

        # 3. Deduplication (simple hash of summary + task)
        # Prevents identical findings from separate subagent runs
        h = hash(f.summary + f.task)
        if h in seen_content_hashes:
            print(f"     Dropped duplicate: {f.task[:40]}...")
            continue

        seen_content_hashes.add(h)
        valid_findings.append(f)

    print(f"  ✅ Kept {len(valid_findings)}/{len(findings)} relevant findings.")

    # Return REPLACEMENT list (Note: requires reducer in schema to handle strict replacement if needed,
    # but since this is a sequential node, it overwrites if we change schema or just clean up here.
    # Actually, standard LangGraph behavior with Annotated[list, add] is APPEND.
    # To FILTER, we usually need to overwrite.
    # Workaround: We pass 'valid_findings' to the next step via a distinct key OR
    # we modify the schema to allow overwrite (standard set).
    # Current schema uses `operator.add`. This is tricky.
    #
    # SOLUTION: We will return a NEW key used by Synthesizer: `filtered_findings`.
    # AND update Synthesizer to look for `filtered_findings` first.
    return {"filtered_findings": valid_findings}


def assign_subagents(state: ResearchState):
    """Fan-out: Assign each task to a subagent OR retry"""
    # Check for retry condition
    if should_retry(state):
        print("  ↺ Routing back to lead_researcher for retry...")
        return "lead_researcher"

    tasks = state.get("subagent_tasks", [])
    print(f"\n🔀 [Fan-out] Distributing {len(tasks)} tasks to subagents...")

    # Send task objects directly
    # CONTEXT ISOLATION: We create a minimal state for the subagent
    # It receives ONLY the query and its specific task.
    # It does NOT receive the full history or other agent findings.
    return [
        Send(
            "subagent",
            {
                "subagent_tasks": [task],
                "query": state["query"],
                "visited_sources": state.get("visited_sources", [])  # Pass unified visited sources
            },
        )
        for task in tasks
    ]


def synthesizer_node(state: SynthesizerState):
    """Synthesizer: Aggregate and synthesize all findings"""
    from prompts import SYNTHESIZER_MAIN, SYNTHESIZER_RETRY, SYNTHESIZER_SYSTEM

    # Prefer filtered findings if available (Guardrail active)
    findings = state.get("filtered_findings", [])
    if not findings:
        # Fallback to raw findings
        findings = state.get("subagent_findings", [])
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")

    print(f"\n📊 [Synthesizer] Synthesizing {len(findings)} findings...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    if not findings:
        return {"synthesized_results": "No findings to synthesize."}

    # Pre-compute findings metadata (token optimization)
    findings_metadata = extract_findings_metadata(findings)

    # Format findings - only include essential information (already optimized)
    findings_text = "\n\n".join([
        f"{i+1}. {f.get('task', 'Unknown')[:60]}\n{f.get('summary', 'No summary')}"
        for i, f in enumerate(findings)
    ])

    # Add metadata context to prompt (helps LLM understand scope without full content)
    metadata_context = (
        f"\n\n[Metadata: {findings_metadata['count']} findings, "
        f"{findings_metadata['total_sources']} sources, "
        f"avg length: {findings_metadata['avg_summary_length']} chars]"
    )

    prompt_content = SYNTHESIZER_MAIN.format(
        query=query,
        findings=findings_text + metadata_context
    )

    if last_error:
        prompt_content = SYNTHESIZER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    structured_llm = get_lead_llm().with_structured_output(
        SynthesisResult, include_raw=True
    )
    response = structured_llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=prompt_content),
    ])

    # Use helper to process retry logic
    def fallback(s):
        return {
            "synthesized_results": (
                "Failed to synthesize findings into structured format."
            )
        }

    retry_state = process_structured_response(response, state, fallback)
    if retry_state:
        # Check if we are clearing the state (success/fallback) or looping (error)
        # Note: process_structured_response returns {error: ...} for loop
        return retry_state

    # Success
    result = response["parsed"]
    print(f"  ✅ Synthesis complete ({len(result.summary)} chars)")

    return {
        "synthesized_results": result.summary,
        "error": None,
        "retry_count": 0
    }


def decision_node(state: DecisionState):
    """Decision: Determine if more research is needed"""
    # Only use metadata, not full content - reduces token usage
    iteration_count = state.get("iteration_count", 0)
    findings_count = len(state.get("subagent_findings", []))
    synthesized_length = len(state.get("synthesized_results", ""))

    print("\n🤔 [Decision] Evaluating if more research is needed...")
    print(
        f"   Iteration: {iteration_count}, "
        f"Findings: {findings_count}, "
        f"Synthesis length: {synthesized_length}"
    )

    # Simple decision logic using only metadata
    # Continue if: iteration < 2 AND (few findings OR short synthesis)
    needs_more = (
        iteration_count < 2 and
        (findings_count < 3 or synthesized_length < 500)
    )

    if needs_more:
        print("  ✅ Decision: Continue research")
        return {"needs_more_research": True}
    else:
        print("  ✅ Decision: Finish research")
        return {"needs_more_research": False}


def citation_agent_node(state: CitationAgentState):
    """CitationAgent: Extract citations and create final report"""
    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")
    query = state["query"]

    print("\n📝 [CitationAgent] Extracting citations and creating final report...")
    print(f"  📊 Processing {len(findings)} findings for citations...")

    # Pre-extract all sources and build lookup map (performance optimization)
    all_source_dicts = []
    for finding in findings:
        # Handle both Pydantic model and dict
        if hasattr(finding, "sources"):
            sources = finding.sources
        else:
            sources = finding.get("sources", [])
        all_source_dicts.extend(sources)

    # Build source metadata map (identifier → title) before loops
    source_map = {}
    for source in all_source_dicts:
        if isinstance(source, dict):
            url = source.get("url", "")
            title = source.get("title", "Unknown")
        else:
            url = getattr(source, "url", "") if hasattr(source, "url") else source.get("url", "") if hasattr(source, "get") else ""
            title = getattr(source, "title", "Unknown") if hasattr(source, "title") else source.get("title", "Unknown") if hasattr(source, "get") else "Unknown"

        if url:
            source_map[url] = title

    # Collect unique citations using pre-built map
    all_sources = []
    seen_urls = set()
    for url, title in source_map.items():
        if url not in seen_urls:
            seen_urls.add(url)
            all_sources.append(Citation(title=title, url=url))
            print(f"      ✅ Added citation: {title[:50]} - {url[:50]}")

    print(f"  📚 Total unique citations collected: {len(all_sources)}")

    # Format citations
    citations = all_sources

    # Create final report - optimized format
    citations_text = "\n".join([
        f"{i+1}. {c.get('title', 'Unknown')} - {c.get('url', '')}"
        for i, c in enumerate(citations)
    ])

    final_report = f"""# Research Report

## Query
{query}

## Findings
{synthesized}

## Sources
{citations_text}
"""

    print(f"  ✅ Final report created with {len(citations)} citations")

    return {
        "citations": citations,
        "final_report": final_report,
    }


def verifier_node(state: ResearchState):
    """Verifier: Cross-check synthesized report against source evidence"""
    from prompts import VERIFIER_MAIN, VERIFIER_SYSTEM
    from schemas import VerificationResult

    report = state.get("synthesized_results", "")
    findings = state.get("subagent_findings", [])

    if not report or not findings:
        return {"final_report": report}

    print("\n🕵️  [Verifier] Cross-checking report against source evidence...")

    # Extract evidence summaries before aggregation (token optimization)
    # Use summaries and metadata instead of full content
    evidence_summaries = extract_evidence_summaries(findings, max_length=500)

    # Aggregate evidence from summaries (much smaller than full content)
    evidence_pieces = []

    # 1. From Findings (Subagent work) - use summaries instead of full content
    for i, f in enumerate(findings):
        if evidence_summaries and i < len(evidence_summaries):
            evidence_pieces.append(evidence_summaries[i])
        elif f.summary:
            # Fallback to summary if extraction failed
            evidence_pieces.append(
                f"Task: {f.task[:50]}\nEvidence: {f.summary[:500]}"
            )

    # 2. From RAG (Direct verification)
    print("  🧠 [Verifier] Retrieving verification context...")
    # Use the first 100 chars of report as proxy query
    verification_query = report[:200]
    rag_evidence, _ = context_manager.retrieve_knowledge(verification_query)
    if rag_evidence:
        evidence_pieces.append(f"Internal Knowledge Base:\n{rag_evidence}")

    if not evidence_pieces:
        print("  ⚠️  No full source text available for verification. Skipping.")
        return {}  # No change

    evidence_text = "\n\n".join(evidence_pieces)

    # Invoke Verifier LLM
    prompt_content = VERIFIER_MAIN.format(
        report=report,
        evidence=evidence_text
    )

    # Use lead LLM (stronger model) for verification
    structured_llm = get_lead_llm().with_structured_output(VerificationResult)

    try:
        response = structured_llm.invoke([
            SystemMessage(content=VERIFIER_SYSTEM),
            HumanMessage(content=prompt_content)
        ])

        if response.is_valid:
            print("  ✅ Report verified: Structure and facts appear accurate.")
            return {}  # No change
        else:
            corrections_count = len(response.corrections)
            print(f"  ⚠️  Issues found. {corrections_count} corrections applied.")
            return {
                "synthesized_results": response.corrected_report
            }

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return {}


def route_decision(
    state: ResearchState,
) -> Literal["lead_researcher", "citation_agent"]:
    """Route based on decision node result"""
    if state.get("needs_more_research", False):
        return "lead_researcher"
    return "citation_agent"


def route_synthesizer(state: ResearchState) -> Literal["synthesizer", "decision"]:
    """Route synthesizer retry or success"""
    if should_retry(state):
        print("  ↺ Routing back to synthesizer for retry...")
        return "synthesizer"
    return "decision"


# Build graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("lead_researcher", lead_researcher_node)
workflow.add_node("subagent", subagent_node)
workflow.add_node("filter_findings", filter_findings_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("decision", decision_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("citation_agent", citation_agent_node)

# Add edges
workflow.add_edge(START, "lead_researcher")
workflow.add_conditional_edges(
    "lead_researcher",
    assign_subagents,
    ["subagent", "lead_researcher"],
)
workflow.add_edge("subagent", "filter_findings")
workflow.add_edge("filter_findings", "synthesizer")
workflow.add_conditional_edges(
    "synthesizer",
    route_synthesizer,
    ["synthesizer", "decision"],
)
workflow.add_conditional_edges(
    "decision",
    route_decision,
    {
        "lead_researcher": "lead_researcher",
        "citation_agent": "verifier",
    },
)
workflow.add_edge("verifier", "citation_agent")
workflow.add_edge("citation_agent", END)

# Compile app
# Compile app with persistence (MemorySaver)
# This enables "Short-Term Memory" (thread-scoped)
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

