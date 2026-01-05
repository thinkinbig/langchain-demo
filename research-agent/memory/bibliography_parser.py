"""
Bibliography Parser
Extracts structured citation graph elements from bibliography text.
"""

from typing import List, Optional

from extraction_service import extract_with_llm
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class CitationGraphElement(BaseModel):
    """Structured representation of a cited paper."""
    title: str = Field(..., description="Title of the cited paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Venue or journal name")


class BibliographyResult(BaseModel):
    """Result of bibliography extraction."""
    citations: List[CitationGraphElement] = Field(default_factory=list)


BIBLIOGRAPHY_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert academic librarian. Your task is to parse a raw bibliography or references section into structured data.

Extract each citation entry into a structured object containing the title, authors, year, and venue.
Normalize author names where possible (e.g., "Smith, J." -> "J. Smith").
Ensure titles are clean and complete.

<bibliography_text>
{text}
</bibliography_text>

Return a JSON object with a 'citations' list.
""")


def parse_bibliography(
    text: str,
    llm: Optional[BaseChatModel] = None
) -> List[CitationGraphElement]:
    """
    Parse a bibliography string into structured citation elements.
    
    Args:
        text: Raw text of the bibliography/references section
        llm: Optional LLM instance
        
    Returns:
        List of CitationGraphElement objects
    """
    if not text or len(text) < 50:
        return []

    # Use the extraction service to handle the heavy lifting
    extracted_dicts = extract_with_llm(
        text=text,
        prompt_template=BIBLIOGRAPHY_EXTRACTION_PROMPT,
        result_schema=BibliographyResult,
        llm=llm
    )
    
    # Convert dicts back to Pydantic models
    results = []
    for d in extracted_dicts:
        try:
            results.append(CitationGraphElement(**d))
        except Exception:
            continue
            
    return results
