"""Analysis node: Analyze gathered context (Internal or Web) and call tools"""

import context_manager
import tools
from context_formatter import ContextFormatter
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_subagent_llm
from memory_helpers import content_metadata_to_string, create_content_metadata
from prompts import (
    SUBAGENT_ANALYSIS_RETRY,
    SUBAGENT_REFINE_WITH_TOOL,
    SUBAGENT_STRUCTURED_ANALYSIS,
    SUBAGENT_SYSTEM,
)
from retrieval import RetrievalResult, RetrievalSource, Source
from schemas import AnalysisOutput, Finding, SubagentState



async def analysis_node(state: SubagentState):
    """Node: Analyze gathered context (Internal or Web) using Structured Flow"""
    import asyncio  # Import locally to avoid global pollution if desired, or move up

    # --- Context Preparation (Same as before) ---
    task_description = state.get("task_description", "")
    if not task_description:
        tasks = state.get("subagent_tasks", [])
        if tasks:
            rs_task = tasks[0]
            task_description = rs_task.description

    # Get retrieval results (unified format)
    internal_result = state.get("internal_result")
    web_result = state.get("web_result")

    # Handle dict serialization (omitted exact logic for brevity,
    # assuming generic hydration works or consistent with prev implementation)
    # Re-hydrating objects if they are dicts
    if internal_result and isinstance(internal_result, dict):
            sources = [Source(**s) for s in internal_result.get("sources", [])]
            internal_result = RetrievalResult(
                content=internal_result.get("content", ""),
                sources=sources,
                source_type=RetrievalSource.INTERNAL,
                has_content=internal_result.get("has_content", False)
            )
    if web_result and isinstance(web_result, dict):
            sources = [Source(**s) for s in web_result.get("sources", [])]
            web_result = RetrievalResult(
                content=web_result.get("content", ""),
                sources=sources,
                source_type=RetrievalSource.WEB,
                has_content=web_result.get("has_content", False)
            )

    visited_sources = state.get("visited_sources", [])
    visited_identifiers = [vs.identifier for vs in visited_sources]

    formatter = ContextFormatter()
    formatted_context, citation_instructions, all_sources = (
        formatter.format_for_analysis(
            task=task_description,
            internal_result=internal_result,
            web_result=web_result,
            visited_identifiers=visited_identifiers
        )
    )

    # Log sources
    print(f"  📚 [Analysis] Analyzing context from {len(all_sources)} sources...")

    # --- Structured Flow ---

    # 1. Prepare Initial Prompt
    initial_prompt = SUBAGENT_STRUCTURED_ANALYSIS.format(
        task=task_description,
        results=formatted_context + citation_instructions
    )
    system_prompt = context_manager.get_system_context(
        SUBAGENT_SYSTEM, include_knowledge=True
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=initial_prompt)
    ]

    llm = get_subagent_llm()
    structured_llm = llm.with_structured_output(AnalysisOutput, include_raw=True)

    # Helper for Retry Logic (Error Prompt Pattern)
    async def invoke_with_retry(msgs, existing_retries=0):
        # Limit local retries to 2
        for _i in range(2):
            try:
                # Async invoke
                response = await structured_llm.ainvoke(msgs)
            except Exception as e:
                # Manually handle Pydantic validation errors that might bubble up
                error_msg = f"Validation Error: {str(e)}"
                print(f"  ⚠️  [Analysis] Exception during invoke: {error_msg}")
                msgs.append(HumanMessage(content=SUBAGENT_ANALYSIS_RETRY.format(
                    previous_prompt="",
                    error=error_msg
                ).content))
                continue

            # Check for parsing error
            if response.get("parsing_error"):
                error_msg = str(response["parsing_error"])
                print(f"  ⚠️  [Analysis] Validation error: {error_msg}")

                # Construct Retry Prompt (Error Prompt Pattern)
                # Maximize KV cache: Append error to history instead of rewriting
                # Actually, standard pattern is to append AI content + Error
                msgs.append(HumanMessage(content=SUBAGENT_ANALYSIS_RETRY.format(
                    previous_prompt="", # Context already in history
                    error=error_msg
                ).content))
                continue

            return response["parsed"]

        # Fallback if retries failed
        print("  ❌ [Analysis] Failed to parse structured output after retries.")
        return AnalysisOutput(
            summary="Analysis failed due to validation errors.",
            citations=[],
            reasoning="Failed validation"
        )

    # 2. Execute Analysis (One-Shot)
    analysis = await invoke_with_retry(messages)

    # 3. Tool Execution Loop (Simulated "Refine" if code present)
    if analysis.python_code:
        print("  🛠️ [Analysis] Executing Python Code...")
        try:
            # We use the python_repl tool function directly
            start_code = analysis.python_code
            
            # Use asyncio.to_thread for blocking tool execution
            result = await asyncio.to_thread(tools.python_repl, start_code)
            
            print(f"     Result: {str(result)[:100]}...")

            # 4. Refine Step (Maximize KV Cache by appending)
            # Append the AI's previous thought (implicitly handled if we kept message
            # history, but here 'analysis' is the structured output. To keep KV cache
            # valid for a *follow-up*, we ideally should have the 'raw' AI message in
            # 'messages'. However, 'with_structured_output' consumes the
            # generation. To strictly support KV cache for the refinement,
            # we append the interaction.

            # Since we can't easily retrieve the exact raw tokens of the previous
            # generation from 'parsed' output in a way that perfectly aligns with
            # server cache (unless using specific providers), we will construct a
            # logical continuation.

            refine_prompt = SUBAGENT_REFINE_WITH_TOOL.format(
                tool_output=result
            )
            messages.append(HumanMessage(content=refine_prompt))

            print("  🔄 [Analysis] Refining with tool output...")
            analysis = await invoke_with_retry(messages)

        except Exception as e:
            print(f"  ⚠️ [Analysis] Code execution failed: {e}")
            # Continue with initial analysis

    # --- Final Output Construction ---

    # Prepare sources list
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

    # Prepare content metadata
    evidence_content = ""
    if formatted_context:
        content_metadata = create_content_metadata(
            formatted_context, max_preview=200
        )
        evidence_content = content_metadata_to_string(content_metadata)

    # Convert schema Citations to dictionaries if needed
    # (Finding expects dictionaries or Citations? Schema says
    # Finding.extracted_citations: List[dict])
    # But ResearchState.citations is List[Citation].
    # Let's align with Finding schema which says List[dict] in schemas.py:42
    # Wait, Finding.extracted_citations is List[dict].
    # AnalysisOutput.citations is List[Citation].
    # We need to convert.
    extracted_citations_dicts = [
        c.model_dump() for c in analysis.citations
    ]

    if extracted_citations_dicts:
        print(f"  📚 [Analysis] Extracted {len(extracted_citations_dicts)} citations")

    finding = Finding(
        task=task_description,
        summary=analysis.summary,
        sources=final_sources,
        content=evidence_content,
        extracted_citations=extracted_citations_dicts
    )

    print("  ✅ [Analysis] Structure flow complete.")
    return {
        "subagent_findings": [finding],
        "extracted_citations": extracted_citations_dicts
    }


