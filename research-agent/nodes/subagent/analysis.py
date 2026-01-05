"""Analysis node: Analyze gathered context (Internal or Web) and call tools"""

import asyncio

import context_manager
import tools
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from nodes.subagent.utils import (
    build_finding_from_analysis,
    invoke_llm_with_retry,
    prepare_analysis_context,
)
from prompts import (
    SUBAGENT_REFINE_WITH_TOOL,
    SUBAGENT_STRUCTURED_ANALYSIS,
    SUBAGENT_SYSTEM,
)
from schemas import AnalysisOutput, SubagentState


async def analysis_node(state: SubagentState):
    """Node: Analyze gathered context (Internal or Web) using Structured Flow"""
    # --- Context Preparation ---
    formatted_context, citation_instructions, all_sources, task_description = (
        prepare_analysis_context(state)
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

    llm = get_llm_by_model_choice("turbo")
    structured_llm = llm.with_structured_output(AnalysisOutput, include_raw=True)

    # 2. Execute Analysis (One-Shot)
    analysis = await invoke_llm_with_retry(structured_llm, messages)

    # 3. Tool Execution Loop (Simulated "Refine" if code present)
    if analysis.python_code:
        print("  🛠️ [Analysis] Executing Python Code...")
        try:
            # We use the python_repl tool function directly
            start_code = analysis.python_code

            # Use asyncio.to_thread for blocking tool execution with timeout
            from config import settings
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tools.python_repl, start_code),
                    timeout=settings.TIMEOUT_PYTHON_REPL
                )
            except asyncio.TimeoutError:
                print(f"  ⚠️  [Analysis] Python code execution timeout "
                      f"({settings.TIMEOUT_PYTHON_REPL}s)")
                result = "(Code execution timed out)"

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
            analysis = await invoke_llm_with_retry(structured_llm, messages)

        except Exception as e:
            print(f"  ⚠️ [Analysis] Code execution failed: {e}")
            # Continue with initial analysis

    # --- Final Output Construction ---
    finding = build_finding_from_analysis(
        analysis=analysis,
        task_description=task_description,
        all_sources=all_sources,
        formatted_context=formatted_context
    )

    extracted_citations_dicts = finding.extracted_citations

    print("  ✅ [Analysis] Structure flow complete.")
    return {
        "subagent_findings": [finding],
        "extracted_citations": extracted_citations_dicts
    }


