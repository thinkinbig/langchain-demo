"""
Extraction Service
Generic service for extracting structured data from text using LLMs.
"""

from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from llm.factory import get_llm_by_model_choice
from pydantic import BaseModel


def extract_with_llm(
    text: str,
    prompt_template: ChatPromptTemplate,
    result_schema: Type[BaseModel],
    llm: Optional[BaseChatModel] = None,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Extract structured data from text using LLM.

    Args:
        text: Text to extract from
        prompt_template: Prompt template to use
        result_schema: Pydantic model for the expected result
        llm: Optional LLM instance (defaults to turbo)
        max_retries: Number of retries for parsing errors

    Returns:
        List of dictionaries containing extracted data
    """
    if not text:
        return []

    if llm is None:
        llm = get_llm_by_model_choice("turbo")

    # Configure LLM with structured output
    structured_llm = llm.with_structured_output(result_schema, include_raw=True)

    # Format prompt
    formatted_prompt = prompt_template.format(text=text)
    messages = [HumanMessage(content=formatted_prompt)]

    # Invoke LLM
    try:
        response = structured_llm.invoke(messages)
        
        if response.get("parsing_error"):
            print(f"  ⚠️  Extraction parsing error: {response['parsing_error']}")
            return []

        parsed = response["parsed"]
        
        # Determine how to return list of dicts based on schema
        # If schema has a list field 'citations' or 'items', return that
        # Otherwise if it's a single item, return [item.dict()]
        
        if hasattr(parsed, "citations"):
            return [c.model_dump() for c in parsed.citations]
        elif hasattr(parsed, "items"):
            return [i.model_dump() for i in parsed.items]
        else:
            return [parsed.model_dump()]

    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return []
