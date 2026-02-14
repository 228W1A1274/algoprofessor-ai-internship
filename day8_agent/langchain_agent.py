"""
LangChain Agent Implementation
Day 8 - Production-Ready Agent with LangChain
"""

import os
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from tools.custom_tools import (
    rag_search,
    calculator,
    analyze_data,
    search_patient,
    check_doctor_availability
)

# Load environment variables
load_dotenv()


def create_hospital_agent():
    """Create a LangChain agent with hospital tools"""
    
    # Initialize LLM
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",

        
        temperature=0.1
    )
    
    # Define tools
    tools = [
        Tool(
            name="HospitalDatabase",
            func=rag_search,
            description="""Use this tool to search the hospital database for information about:
            - Doctors and their specializations
            - Patients and their medical records
            - Appointments and schedules
            - Departments and facilities
            - Medical procedures and services
            Input should be a search query describing what you want to find."""
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="""Use this tool for mathematical calculations such as:
            - Adding costs or fees
            - Calculating totals, averages, percentages
            - Any arithmetic operations
            Input should be a mathematical expression like '150 + 200' or '(500 - 350) / 500 * 100'"""
        ),
        Tool(
            name="DataAnalysis",
            func=analyze_data,
            description="""Use this tool to analyze hospital data and get statistics:
            - 'occupancy' for department occupancy rates
            - 'costs' for medical procedure cost analysis
            - 'doctors' for doctor statistics
            Input should be the type of analysis needed."""
        ),
        Tool(
            name="PatientSearch",
            func=search_patient,
            description="""Use this tool to search for a specific patient by name.
            Returns detailed patient information including diagnosis, room number, and attending physician.
            Input should be the patient's name."""
        ),
        Tool(
            name="DoctorAvailability",
            func=check_doctor_availability,
            description="""Use this tool to check which doctors are available.
            You can filter by specialization (cardiology, neurology, pediatrics, orthopedics)
            or leave empty to see all doctors.
            Input should be the specialization or empty string for all doctors."""
        )
    ]
    
    # Create prompt template for ReAct agent
    template = """You are a helpful AI assistant for City General Hospital. 
    
Answer questions about hospital operations, doctors, patients, appointments, and medical services.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    return agent_executor


def main():
    """Test the LangChain agent"""
    
    print("\n" + "="*60)
    print("HOSPITAL AI AGENT - LangChain Implementation")
    print("="*60 + "\n")
    
    # Create agent
    agent = create_hospital_agent()
    
    # Test queries
    test_queries = [
        "What is the total consultation fee if I visit Dr. Sarah Johnson and Dr. James Wilson?",
        "Show me all doctors specializing in cardiology",
        "What is the average occupancy rate across all departments?",
        "Find patient John Doe and tell me which doctor is treating him",
        "How much would it cost for an MRI scan and an X-ray combined?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*60}\n")
        
        try:
            result = agent.invoke({"input": query})
            print(f"\n📊 FINAL ANSWER:\n{result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
        
        if i < len(test_queries):
            input("Press Enter for next query...")


if __name__ == "__main__":
    main()