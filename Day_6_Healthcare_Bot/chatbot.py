from langchain_core.output_parsers import StrOutputParser
from llm_integration import get_llm
from prompt_templates import get_healthcare_prompt

def setup_chain():
    """
    Connects the Prompt -> LLM -> Output Parser
    """
    llm = get_llm()
    prompt = get_healthcare_prompt()

    # LCEL (LangChain Expression Language) Pipeline
    chain = prompt | llm | StrOutputParser()
    
    return chain